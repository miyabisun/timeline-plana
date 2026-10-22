# Timeline Plana

## Overview

これより、先生の戦闘をサポートいたします。

timeline-aronaで作成したタイムラインファイルを読み込ませてください。<br>
画面上に表示されている戦闘画面の残り時間やコストの情報を元に、<br>
生徒のEXスキルの発動タイミングをディスプレイに書き加えてアシストします。

司令塔としての先生も一人ではありません。<br>
これからは私たちも先生と共に戦います。

## FAQ

Q: 「Timeline Plana」ってどういう意味なの？<br>
A: "タイムライン"はブルーアーカイブの用語で、ボス戦攻略の鍵<del>(ケイ)</del>となるレシピです。<br>
しかし、ボス戦においてタイムラインの完遂も重要な要素。<br>
出来の良い妹がタイムライン通りの攻略をサポートしてくれますようにという願いが込められています。

Q: 先生！なぜこのアロナちゃんを差し置いて「Timeline Plana」なのでしょうか？<br>
A: <del>最終編のアロナの語彙力のなさにはガッカリだよ</del><br>
"戦闘にはタイムラインも重要だからね"<br>
"これから一緒に「Timeline Arona」でタイムラインを作成しようか"<br>
"アロナはお姉さんだから戦闘支援の方はプラナ主体でお願いするね？"

## 必要なもの

- Windows 10 / 11（PC版ブルーアーカイブ `BlueArchive.exe` を画面キャプチャします）
- [timeline-arona](https://github.com/miyabisun/timeline-arona) の形式で書いたタイムラインの YAML ファイル（形式は timeline-arona の [docs/README.md](https://github.com/miyabisun/timeline-arona/blob/main/docs/README.md)）

## 導入

1. [Releases](https://github.com/miyabisun/timeline-plana/releases) の最新版から `timeline-plana-x86_64-pc-windows-msvc.exe` を取得します。
   同じ版の `.sha256` で、ファイルが壊れていないかを確かめられます（PowerShell の `Get-FileHash <exe>` の値と比べます）。
2. 好きなフォルダーに置いて起動します。インストーラーはなく、管理者権限も要りません。

更新するときは、新しい版の exe を同じ場所に置き換えます。

## 使い方

1. ブルーアーカイブを起動してから Timeline Plana を起動します。**Connection** タブがゲームの窓を見つけて接続し、残り時間・コスト・戦闘の状態を読み始めます。
2. **Timeline** タブの **Open YAML** で、タイムラインのファイルを開きます。誤りがあれば、どの行の何が違うかが表示されます。
3. **Show overlay** を押すと、ゲームの窓の右端に、次に使う EX スキルの組とそのきっかけ（残り時間かコスト）が重なって表示されます。使う時になると **NOW** に変わり、組の中で次に押すスキルに ▶ が付きます。近づいた時間マーカー（`time` の行）も表示されます。

### 案内の進み方

- Timeline Plana は押したカードを直接は見られないため、戦闘中に EX コストが 1 近く減ったところで「スキルを使った」と数え、次のスキルへ進みます。ポーズ中やスキル選択のスロー中は進みません。
- 残り時間が戻ったとき（新しい戦闘）は、タイムラインの先頭からやり直します。
- 読み違いなどで案内がずれたときは、Timeline タブで続けたいスキルの行を押すと、そこから案内し直します。

## 開発

```sh
npm ci
npm run tauri dev        # 開発用に起動
npm run verify           # build・型・e2e・rustfmt・clippy・テスト
git config core.hooksPath .githooks   # clone ごとに一度: main とタグの push 前に verify を実行
```

リリースは `vX.Y.Z` のタグを push すると GitHub Actions が exe と `.sha256` を作ります（`.github/workflows/release.yml`。タグは `package.json` の version と一致させます）。
