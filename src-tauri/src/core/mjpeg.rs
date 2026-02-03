use std::io::Write;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use tiny_http::{Header, Response, Server, StatusCode};

pub struct MjpegState {
    pub current_frame: Mutex<Option<Vec<u8>>>,
    pub cond: Condvar,
}

impl MjpegState {
    pub fn new() -> Self {
        Self {
            current_frame: Mutex::new(None),
            cond: Condvar::new(),
        }
    }

    pub fn update_frame(&self, frame: Vec<u8>) {
        let mut lock = self.current_frame.lock().unwrap();
        *lock = Some(frame);
        self.cond.notify_all();
    }
}

pub fn start_server(state: Arc<MjpegState>, port: u16) {
    thread::spawn(move || {
        let server =
            Server::http(format!("0.0.0.0:{}", port)).expect("Failed to start MJPEG server");
        println!("MJPEG Server listening on port {}", port);

        for request in server.incoming_requests() {
            let state_clone = state.clone();
            thread::spawn(move || {
                handle_client(request, state_clone);
            });
        }
    });
}

fn handle_client(request: tiny_http::Request, state: Arc<MjpegState>) {
    let boundary = "boundary";
    let headers = vec![
        Header::from_bytes(
            &b"Content-Type"[..],
            &b"multipart/x-mixed-replace; boundary=boundary"[..],
        )
        .unwrap(),
        Header::from_bytes(&b"Connection"[..], &b"close"[..]).unwrap(),
        Header::from_bytes(&b"Access-Control-Allow-Origin"[..], &b"*"[..]).unwrap(),
    ];

    let mut response = Response::new_empty(StatusCode(200));
    for h in headers {
        response.add_header(h);
    }

    // Hijack the writer to stream manually
    let mut writer = request.into_writer();

    // Write initial headers
    let header_str = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary={}\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
        boundary
    );
    if writer.write_all(header_str.as_bytes()).is_err() {
        return;
    }

    let mut last_frame: Option<Vec<u8>> = None;

    loop {
        let frame_data = {
            let mut lock = state.current_frame.lock().unwrap();

            // If no frame yet, wait.
            // If we have a frame, check if it's new?
            // MJPEG usually just streams current state.
            // Ideally we wait for *next* frame.
            // But Condition Variable wait_timeout is good for heartbeats.
            // For now, simple wait.
            // To avoid sending same frame twice instantly, we could track IDs or just wait for notify.
            // But notify_all happens on Update.
            // So we wait.
            lock = state.cond.wait(lock).unwrap();
            lock.clone()
        };

        if let Some(data) = frame_data {
            let part_headers = format!(
                "--{}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                boundary,
                data.len()
            );

            if writer.write_all(part_headers.as_bytes()).is_err() {
                break;
            }
            if writer.write_all(&data).is_err() {
                break;
            }
            if writer.write_all(b"\r\n").is_err() {
                break;
            }
            // flush? tiny_http buffer might need it depending on impl, usually write_all pushes enough.
            // but explicit flush is good.
            if writer.flush().is_err() {
                break;
            }
        }
    }
}
