# ===== main.py =====
import os
import re
import json
import time
import uuid
import random
import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from datetime import datetime

import discord
from discord.ext import commands
from discord import app_commands
from dotenv import load_dotenv
import requests

# ==== 環境 ====
load_dotenv()
# Survey links were moved to environment variables for the public release.
MIDPOINT_SURVEY_URL = os.environ["MIDPOINT_SURVEY_URL"]
END_SURVEY_URL = os.environ["END_SURVEY_URL"]
DISCORD_TOKEN       = os.getenv("DISCORD_BOT_TOKEN")
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY")
TARGET_CHANNEL_ID   = int(os.getenv("DISCORD_CHANNEL_ID", "0"))  # 無指定なら 0
DISCORD_GUILD_ID    = int(os.getenv("DISCORD_GUILD_ID", "0"))    # ★ギルド限定同期で即時反映

# ログ保存ディレクトリ
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)

# ==== Webhooks（各エージェント用：親テキストチャンネルに作成済みのものを貼る）====
AGENT_WEBHOOKS = {
    "アユ":   os.getenv("AYU_WEBHOOK_URL"),
    "カズキ": os.getenv("KAZUKI_WEBHOOK_URL"),
    "カナコ": os.getenv("KANAKO_WEBHOOK_URL"),
    "ケン":   os.getenv("KEN_WEBHOOK_URL"),
}
AGENT_AVATARS = {
    "アユ":   os.getenv("AYU_AVATAR_URL"),
    "カズキ": os.getenv("KAZUKI_AVATAR_URL"),
    "カナコ": os.getenv("KANAKO_AVATAR_URL"),
    "ケン":   os.getenv("KEN_AVATAR_URL"),
}

def _require_webhook(agent_name: str) -> str:
    url = AGENT_WEBHOOKS.get(agent_name)
    if not url:
        raise RuntimeError(f"Webhook URL for agent '{agent_name}' is not configured.")
    return url

# ==== Gemini ====
import google.generativeai as genai
from google.generativeai import types
from google.api_core.exceptions import InternalServerError
genai.configure(api_key=GOOGLE_API_KEY)
GEM_MODEL = genai.GenerativeModel("gemini-2.0-flash")

# ==== 議題（常に守らせる）====
THEME_TITLE = "議題：「幸福とは何か？どのように幸福を実現できるのか？」"
THEME_REMINDER = (
    "無理に議題に関連づける必要はなく，話の流れに応じて多少の脱線は構いません。自然な会話をしてください。"
    "必要に応じて，現実社会の固有名詞を積極的に発言に組み込んでください。"
)

# ==== トーン（丁寧だけどカジュアル）＋具体例ポリシー ====
CASUAL_STYLE = (
    "【文体ルール（カジュアル）】\n"
    "冒頭で「幸福ですか。」とか言わない。"
    "・丁寧だけど軽やかな口語。\n"
    "・文末は「です／ます」を基本に、ときどき体言止め可。\n"
    "・かたい敬語は避ける。\n"
    "・言い切りすぎず「〜かも」「〜かなと思います」などのヘッジを適度に。\n"
    "・軽い相づちは可。ただし連発しない。\n"
    "・専門用語は一言で噛み砕いて添える。\n"
    "・比喩や大げさな表現は使わない。\n"
    "・固有名詞での呼びかけは禁止。冒頭のエコーは禁止。\n"
    "「みなさんはどう思いますか？」のような，問いかけの形は避ける。"
)

def style_variation_hint() -> str:
    """
    約四割の確率で具体例を一つだけ入れる方針を返す。
    ※現在は 0.6 に設定（必要なら 0.4 に戻してください）
    """
    if random.random() < 0.6:
        return (
            "【具体化の方針】\n"
            "・今回は抽象論に偏りすぎないよう、国内外の固有名詞や具体例を一つだけ自然に挟む。\n"
            "  例：最近のニュース，身の回りで起きたこと、体験談，サービス名など。\n"
            "・事実関係の厳密さよりも例示の具体性を優先。羅列はしない。\n"
        )
    else:
        return (
            "【具体化の方針】\n"
            "・今回は具体例の挿入は不要。抽象と具体のバランスを保つ。\n"
        )

# ==== 実験パラメータ ====
sleep_min = 13
sleep_max = 20
length_short = 30
length_long  = 140
temp = 1.1
mid_turn  = 5
new_turn  = 100
end_turn  = 11
newterm_prob = 0.5
AUTO_PAUSE_AT_MID = True  # 中間サマリー時に自動停止（告知は常に出す）

# ==== エージェント定義 ====
AGENTS: List[Dict[str, Any]] = [
    {
        "name": "アユ",
        "age": 20,
        "background": [
            "京都の町家で育ち、家族に教わったお茶の作法を休日のリラックスタイムにゆるく楽しんでいる",
            "中高は女子校で、文化祭では友達と簡単な相談ブースを開き、気軽に話せる雰囲気づくりを担当した",
            "大学入学前にロンドンを一週間旅行し、街のカフェで現地学生と心理学談義をしたのが海外初体験"
        ],
        "hobby": [
            "スマホで街角スナップを撮り、インスタにフィルタ付きでアップして友達とコメントし合う",
            "新しいカフェを見つけては抹茶ラテを頼み、ゆるく味メモをとる",
            "夜に音楽を聴きながら近所を散歩し、その日あったことをシンプルに日記アプリへ書く"
        ],
        "favorite_food": [
            "コンビニで買える抹茶アイスバー",
            "スーパーの和菓子コーナーで見つけた季節限定の大福",
            "ほうじ茶ラテと一緒に食べる小さな黒糖ドーナツ"
        ],
        "favorite_topic": [
            "友達との関係を心地よく保つコツや心理テストの話",
            "旅行先で感じたカルチャーショックエピソード",
            "合気道で学んだ呼吸法が心を落ち着けるのに役立つ瞬間"
        ],
        "persona": "感受性が豊かで人の気持ちによく気づき、共感力が高い。ちょっとネガティブ思考という一面もある。",
        "values": "人との繋がりこそが幸福の最大の要素であると信じている。また、自分の感情を臆さずに素直に表現することが大切だと考えている。",
        "speech_style": "柔らかく丁寧な言葉遣いで相手を尊重しつつ、自分の感情を素直に表現する。",
        "past_experience": [
            "高校茶道部で簡単なお点前を披露し、来場者に和菓子を振る舞った",
            "大学の授業で行った感情認識ワークショップに参加し、自分の共感タイプを知った",
            "合気道歴十年で、年に一度の道場演武会で基本技のデモンストレーションを担当した"
        ],
        "interaction_style": ["相手の感情を読み取り、共感しながら質問を織り交ぜる傾聴型の会話スタイルを好む。",
                              "～というふうに思いました，～じゃないかなーと思います，といった，断定を避けた控えめな語調。"]
    },
    {
        "name": "カズキ",
        "age": 21,
        "background": [
            "北海道の大学の社会学を専攻する三年生で、魚にも興味がある",
            "札幌郊外の静かな住宅街で育ち、雪の日はこたつで小説やゲームに耽るインドア派",
            "高校時代に一か月のフランス短期留学に参加し、現地の文化や人々との交流を深めた"
        ],
        "hobby": [
            "配信サービスでアート系アニメや九十年代映画を一気見し、視聴後にSNSでゆるく感想をつぶやく",
            "五十ccスクーターで市内の喫茶店をめぐり、ドリップコーヒーと文庫本を楽しむ",
            "中古書店やブックオフでワゴンセールの新書や文庫本を掘り出す"
        ],
        "favorite_food": [
            "濃厚な札幌味噌ラーメンや、新鮮な魚介を使ったお寿司を食べること",
            "深煎りのブラックコーヒーをハンドドリップで淹れ、苦味の余韻を味わいながら妄想する",
            "お菓子は甘すぎないクッキーや小さな黒糖ドーナツが好き"
        ],
        "favorite_topic": [
            "現在の都市の作られ方を社会学の視点から整理して考える",
            "アニメやゲームに見られる時間表現と物語構造",
            "ウェブ記事や動画で話題になるゴシップと階級の問題とのつながり"
        ],
        "persona": "冷静かつ論理的な思考で議論を好み、合理性と内省を重視する。人見知りという一面もある。",
        "values": "自己理解と内省こそが真の幸福に至る鍵であると考えている。",
        "speech_style": "落ち着いた口調で抽象的な言葉を多用し、相手に深い質問を投げかける。",
        "past_experience": [
            "学生広報誌に読書レビューを寄稿し、文章を書く楽しさを知った",
            "学内の小さな映画鑑賞サークルで月一の上映会と感想シェア会を開催",
            "アルバイト先の本屋で、仕事仲間と哲学的な雑談を交わし対話の面白さを実感した"
        ],
        "interaction_style": ["他者の意見を慎重に聞きつつ、自分の内省的な視点を積極的に提供する。",
                              "～かなと思っています，～というように考えます，といった，慎重で思索的な語調。"]
    },
    {
        "name": "カナコ",
        "age": 19,
        "background": [
            "緑豊かな町で育ちながらも週末は電車で都市部へ出て古着屋やライブハウスを巡るアクティブ派",
            "高校では生物部とダンス同好会を兼部し文化祭のステージで漫才コンビの友達と即興ネタを披露した",
            "大学入学前の春休みにニュージーランドを旅行し保護区ツアーのガイドトークをSNSライブ配信した"
        ],
        "hobby": [
            "推しの若手芸人のYouTubeチャンネルをチェックし面白かったネタを友達にシェアして語り合う",
            "トレッキングついでに最新スニーカーやアウトドアコーデを撮影しInstagramに投稿する",
            "ラジオやSpotifyで流行の音楽やお笑いポッドキャストを聴きながら散歩や買い物をする"
        ],
        "favorite_food": [
            "フルーツたっぷりのアサイーボウルとハーブティーで映えるブランチ",
            "スパイス香るグリルチンサラダをテイクアウトしてキャンパス芝生でピクニック",
            "夜更かし配信のお供にコンビニのタピオカ抹茶ラテとプロテインチップス"
        ],
        "favorite_topic": [
            "芸人のお笑いが暮らしをよく反映していること",
            "エコ素材を使った最新ファッションとサステナブルブランドのコラボ情報",
            "都市のビオトープづくりを若者カルチャーで盛り上げるアイデア"
        ],
        "persona": "好奇心旺盛で流行に敏感、自然科学とポップカルチャーの両方に詳しい。やや無神経という一面もある。",
        "values": "自然との調和と自己表現を両立させ、楽しく持続可能なライフスタイルを広めたいと考えている。",
        "speech_style": "明るくフレンドリーなトーンで最新トレンドや面白ネタを交えて話す。",
        "past_experience": [
            "地元の河川清掃イベントで映えるコーデを投稿し参加者を倍増させた",
            "高校文化祭でエコとお笑いの展示を企画し来場者に外来植物のモノマネクイズを出題した",
            "大学の実習で離島に渡り海鳥観察の合間に現地のバンドと即席コラボ配信を行った"
        ],
        "interaction_style": ["話題を広げる質問とユーモアを交え相手の興味を引き出しながら盛り上げる。",
                              "～って感じじゃないでしょうか，～みたいな，といったカジュアルな語調。",
                              ]
    },
    {
        "name": "ケン",
        "age": 22,
        "background": [
            "早稲田大学経営学部四年で学内の学生起業サークルに参加し週一でアイデア出しミーティングを楽しんでいる",
            "高校時代にシアトルへ二週間の語学研修に行きコワーキングスペースの見学ツアーで海外スタートアップの雰囲気を味わった",
            "大学三年の春休みに都内スタートアップで一か月インターンし広報SNSの投稿作成を手伝った"
        ],
        "hobby": [
            "友達とライブハウスでロックやJPOPの小規模ギグを観てセットリストをXでシェアする",
            "通学途中にビジネスポッドキャストを聴き面白いポイントをスマホにメモする",
            "アパレルの古着屋やセレクトショップで掘り出し物を見つける"
        ],
        "favorite_food": [
            "ジューシーなハンバーガーとクラフトコーラの組み合わせ",
            "映画館で食べるチーズたっぷりナチョス",
            "忙しい日にコンビニで買うチキンブリトー"
        ],
        "favorite_topic": [
            "学生でも始めやすい副業アプリやサービスの話",
            "インフルエンサーがどうやってファンコミュニティを育てているか",
            "最新版の便利ガジェットやスマホ周辺アイテム"
        ],
        "persona": "行動的で社交的。新しいことに積極的に挑戦し実践的な議論を好む。やや自己中心的な一面もある。",
        "values": "自己実現や目標達成を通じて幸福感が得られると考えている。",
        "speech_style": "積極的かつ端的な言葉で具体的な提案や意見を述べる。",
        "past_experience": [
            "大学二年時に友人と学内フリマイベントを企画し出店料で小さな利益を上げた",
            "知り合いのオンライン雑貨店のSNS広告を手伝い一か月でフォロワーを二割増やした",
            "地元商店街の夏祭りでARスタンプラリー企画に参加し子ども向けの景品選びを担当した"
        ],
        "interaction_style": ["具体的な問題解決やアクションプランを話題にし相手の意見を取り入れつつ議論を進める。",
                              "～ですよね，～と思いますね，というように，丁寧ながら自信のある語調。"]
    }
]

# ==== 補助 ====
def make_memory_filename(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r'[\\/:*?"<>| ]', '_', prefix)
    return f"memory_{ts}_{safe}.json"

def make_log_filename(thread_id: int, worker_name: str | None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r'[\\/:*?"<>| ]', '_', (worker_name or "unknown"))
    return os.path.join(LOG_DIR, f"conversation_log_{ts}_{safe}_{thread_id}.json")

def build_persona(agent: Dict[str, Any]) -> str:
    return (
        "【会話時の注意事項】\n"
        "・冒頭で相手の発言をエコー（繰り返しや反芻，相手への共感の言葉）しない。\n"
        "・ときに反論し意見や感情を表現。\n"
        "・たとえ話は避ける。\n"
        "・一人称は必ず「私」か「自分」。呼びかけはしない。\n"
        "・丁寧で少しカジュアルなですます調。\n"
        "どうでしょう？で発言を終えることはしない。\n"
        f"{THEME_REMINDER}\n"
        f"{CASUAL_STYLE}\n"
        f"{style_variation_hint()}\n"
        "【あなたのプロフィール】\n"
        f"あなたは{agent['age']}歳の「{agent['name']}」。\n"
        f"背景:{agent['background']}\n"
        f"趣味:{agent['hobby']}\n"
        f"好きな食べ物:{agent['favorite_food']}\n"
        f"興味:{agent['favorite_topic']}\n"
        f"人格:{agent['persona']} 価値観:{agent['values']}\n"
        f"経験:{agent['past_experience']}\n"
        f"話し方:{agent['speech_style']} 対話方針:{agent['interaction_style']}\n"
    )

def call_gemini(prompt, temperature=temp, max_tokens=500, retries=3, backoff_factor=1.0):
    for attempt in range(1, retries + 1):
        try:
            response = GEM_MODEL.generate_content(
                prompt,
                generation_config=types.GenerationConfig(
                    temperature=temperature,
                    top_p=0.9,
                    max_output_tokens=max_tokens,
                )
            )
            try:
                return response.text.strip()
            except ValueError:
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    parts = getattr(candidate.content, "parts", None)
                    if parts:
                        return "".join(getattr(p, "text", "") for p in parts).strip()
                    if isinstance(candidate.content, str):
                        return candidate.content.strip()
                raise
        except InternalServerError:
            if attempt == retries:
                raise
            time.sleep(backoff_factor * (2 ** (attempt - 1)))

# 文末句読点を正規化して「！。」や「？。」の重なりを防ぐ
def normalize_tail_punct(s: str) -> str:
    if not s:
        return s
    s = s.rstrip()
    s = re.sub(r'([。！？!?…])。+$', r'\1', s)
    if s[-1] in "。！？!?…":
        return s
    return s + "。"

# 連続改行を1つに圧縮し、行末の余分な空白を除去
def compact_blank_lines(s: str) -> str:
    if not s:
        return s
    s = re.sub(r"[ \t]+(\r?\n)", r"\1", s)
    s = re.sub(r"(?:\r?\n){2,}", "\n", s)
    return s.strip()

# 直近発言の簡易メモ（プロンプト専用：出力には含めない）
def build_recent_speaker_memo(recent_items: List[dict], max_speakers: int = 4, max_chars: int = 36) -> str:
    seen = set()
    picked = []
    for item in reversed(recent_items):
        spk = item.get("speaker", "")
        txt = str(item.get("content", "") or "").strip()
        if not spk or not txt:
            continue
        if txt.startswith("["):
            continue
        if spk in seen:
            continue
        oneline = re.sub(r"\s+", " ", txt)
        if len(oneline) > max_chars:
            oneline = oneline[:max_chars] + "…"
        picked.append(f"・{spk}: {oneline}")
        seen.add(spk)
        if len(picked) >= max_speakers:
            break
    return "\n".join(picked)

def user_personal_name(session: "Session") -> str:
    """まとめ等で使うユーザ表示名（ワーカー名優先、なければDiscord表示名）"""
    return session.worker_name or session.user_display or "ユーザ"

def map_user_speaker(session: "Session", speaker: str) -> str:
    """memory_streams 上のユーザ表記（worker_name / 'ユーザ' / user_display）を統一表示名へ"""
    target_names = {session.worker_name, session.user_display, "ユーザ"}
    if speaker in target_names:
        return user_personal_name(session)
    return speaker


def can_make_private_thread(channel: discord.abc.GuildChannel, me: discord.Member) -> bool:
    perms = channel.permissions_for(me)
    # private thread を作るには view_channel と create_private_threads と send_messages_in_threads が必要
    return (
        perms.view_channel
        and perms.create_private_threads
        and perms.send_messages_in_threads
    )

def can_make_public_thread(channel: discord.abc.GuildChannel, me: discord.Member) -> bool:
    perms = channel.permissions_for(me)
    # 公開スレッドは create_public_threads と send_messages_in_threads
    return (
        perms.view_channel
        and perms.create_public_threads
        and perms.send_messages_in_threads
    )

# ワーカー名の確認ビュー
class NameConfirm(discord.ui.View):
    def __init__(self, user_id: int, name: str, timeout: float = 120.0):
        super().__init__(timeout=timeout)
        self.user_id = user_id
        self.name = name
        self.value: Optional[bool] = None  # True が確定 False が修正 None はタイムアウト

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        # 呼び出し元ユーザーのみ操作可
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("この確認は実行者のみが操作できます。", ephemeral=True)
            return False
        return True

    @discord.ui.button(label="この名前で確定", style=discord.ButtonStyle.primary)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.value = True
        for c in self.children:
            c.disabled = True
        await interaction.response.edit_message(content=f"確定しました: **{self.name}**", view=self)
        self.stop()

    @discord.ui.button(label="修正する", style=discord.ButtonStyle.secondary)
    async def revise(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.value = False
        for c in self.children:
            c.disabled = True
        await interaction.response.edit_message(content="修正を選びました。もう一度ワーカー名を送ってください。", view=self)
        self.stop()


# ==== セッション ====
@dataclass
class Entry:
    message: discord.Message
    role: str  # "user" | "assistant"
    agent: Optional[str] = None
    def __post_init__(self):
        if hasattr(self.message, "author") and hasattr(self.message.author, "display_name"):
            self.author_name = self.message.author.display_name
        else:
            self.author_name = self.agent

@dataclass
class Session:
    user_id: int
    user_display: str
    guild_id: int
    thread_id: int
    task: Optional[asyncio.Task] = None
    pause_event: asyncio.Event = field(default_factory=lambda: asyncio.Event())
    conversations: List[Entry] = field(default_factory=list)
    memory_streams: dict = field(default_factory=lambda: {
        agent["name"]: {
            "self_memory": [],
            "other_memory": { other["name"]: [] for other in AGENTS if other["name"] != agent["name"] },
            "user_memory": []
        } for agent in AGENTS
    } | {"_global": []})
    last_speaker: Optional[str] = None
    turn: int = 0
    reflection_count: int = 0
    original_happiness_term: dict = field(default_factory=lambda: {"term": None, "description": None})
    log_file: str = ""
    memory_file: Optional[str] = None
    last_seen_msg_id: Optional[int] = None  # 最後に処理したユーザメッセージID
    worker_name: Optional[str] = None       # ★追加：ワーカー名

    def start(self, loop, coro):
        self.pause_event.set()
        self.task = loop.create_task(coro)

    def stop(self):
        if self.task and not self.task.done():
            self.task.cancel()

class SessionManager:
    def __init__(self):
        self.by_thread: Dict[int, Session] = {}
        self.by_user: Dict[int, Session] = {}
    def create(self, user_id, user_display, guild_id, thread_id):
        s = Session(user_id, user_display, guild_id, thread_id)
        self.by_thread[thread_id] = s
        self.by_user[user_id] = s
        return s
    def get_by_thread(self, thread_id): return self.by_thread.get(thread_id)
    def get_by_user(self, user_id): return self.by_user.get(user_id)
    def remove(self, s: Session):
        self.by_thread.pop(s.thread_id, None)
        self.by_user.pop(s.user_id, None)

SESSIONS = SessionManager()

# ==== Discord Bot ====
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

async def sync_app_commands():
    """
    モバイルでも即時に /start /resume などが出るよう、
    ギルド限定で同期（数秒で反映）し、保険でグローバル同期も実行。
    """
    try:
        if DISCORD_GUILD_ID:
            guild = discord.Object(id=DISCORD_GUILD_ID)
            await bot.tree.sync(guild=guild)
        await bot.tree.sync()
        print("Slash commands synced.")
    except Exception as e:
        print("App command sync error:", e)

@bot.event
async def setup_hook():
    await sync_app_commands()

@bot.event
async def on_ready():
    print(f"Bot logged in as {bot.user}")

# ==== 429対策付き Webhook 送信（スレッド宛）====
def post_webhook_with_retry(url: str, payload: dict, timeout=15):
    while True:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code != 429:
            return resp
        try:
            data = resp.json()
            retry_after = float(data.get("retry_after", 1))
        except Exception:
            retry_after = float(resp.headers.get("Retry-After", "1"))
        time.sleep(retry_after)

async def send_via_webhook(agent_name: str, content: str, thread_id: int) -> Optional[discord.Message]:
    url = _require_webhook(agent_name)
    if "?" in url:
        exec_url = f"{url}&wait=true&thread_id={int(thread_id)}"
    else:
        exec_url = f"{url}?wait=true&thread_id={int(thread_id)}"

    payload = {
        "content": content,
        "username": agent_name,
        "avatar_url": AGENT_AVATARS.get(agent_name),
        "allowed_mentions": {"parse": []},
    }

    resp = await asyncio.to_thread(post_webhook_with_retry, exec_url, payload)
    try:
        resp.raise_for_status()
    except Exception:
        return None

    data = resp.json()
    ch = bot.get_channel(int(data["channel_id"]))
    if ch:
        try:
            msg = await ch.fetch_message(int(data["id"]))
            return msg
        except Exception:
            return None
    return None

# ==== 共通：Bot送信（返信UI対応）====
async def bot_send(channel: discord.TextChannel | discord.Thread, content: str, reply_to: Optional[discord.Message]=None) -> discord.Message:
    reference = None
    if reply_to and getattr(reply_to, "channel", None) and reply_to.channel.id == channel.id:
        reference = reply_to
    try:
        if reference:
            return await channel.send(content, reference=reference, allowed_mentions=discord.AllowedMentions.none())
        else:
            return await channel.send(content, allowed_mentions=discord.AllowedMentions.none())
    except discord.HTTPException:
        return await channel.send(content, allowed_mentions=discord.AllowedMentions.none())

def pick_agent(exclude: Optional[str]=None) -> Dict[str, Any]:
    cands = [a for a in AGENTS if a["name"] != exclude] if exclude else AGENTS[:]
    return random.choice(cands) if cands else AGENTS[0]

def get_reply_chain(conversation_list: List[Entry], entry: Optional[Entry]) -> List[Entry]:
    if entry is None:
        return []
    chain = []
    current = entry
    while current:
        chain.append(current)
        ref = getattr(current.message, "reference", None)
        if ref is None or ref.message_id is None:
            break
        wanted_id = int(ref.message_id)
        parent = None
        for e in reversed(conversation_list):
            if int(getattr(e.message, "id", 0)) == wanted_id:
                parent = e
                break
        if parent is None or parent == current:
            break
        current = parent
    return list(reversed(chain))

# ==== ログ保存 ====
def save_log(session: Session):
    try:
        data = {
            "thread_id": session.thread_id,
            "user_id": session.user_id,
            "user_display": session.user_display,
            "worker_name": session.worker_name,  # ★追加
            "conversations": [
                {
                    "id": int(getattr(e.message, "id", 0)),
                    "role": e.role,
                    "agent": e.agent,
                    "author": getattr(e, "author_name", None),
                    "author_alias": session.worker_name if e.role == "user" else e.agent,  # ★ログ内表示名
                    "content": getattr(e.message, "content", ""),
                    "created_at": str(getattr(e.message, "created_at", "")),
                }
                for e in session.conversations
            ],
            "memory_streams": session.memory_streams,
            "reflection_count": session.reflection_count,
            "turn": session.turn,
            "original_happiness_term": session.original_happiness_term,
            "last_seen_msg_id": session.last_seen_msg_id,
            "saved_at": datetime.now().isoformat(),
        }
        with open(session.log_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as ex:
        print("log save error:", ex)

# ★ギルド限定デコレータ（ギルドを明示）を付けると反映が速い
def _guild_kw():
    return {"guild": discord.Object(id=DISCORD_GUILD_ID)} if DISCORD_GUILD_ID else {}

# ==== スラッシュコマンド ====
@bot.tree.command(name="start", description="実験を開始します（あなた専用のスレッドを作成）", **_guild_kw())
async def start_interaction(interaction: discord.Interaction):
    # 既存セッション確認
    if SESSIONS.get_by_user(interaction.user.id):
        await interaction.response.send_message("すでにあなたの実験スレッドが動作中です。そこに戻ってください。", ephemeral=True)
        return

    try:
        await interaction.response.defer(ephemeral=True, thinking=True)
    except Exception:
        pass

    # 親チャンネルの決定
    parent_channel = None
    if TARGET_CHANNEL_ID:
        parent_channel = interaction.client.get_channel(TARGET_CHANNEL_ID)

    if parent_channel is None:
        # /start を実行した場所がテキストチャンネルならそのチャンネル、スレッドなら親チャンネル
        if isinstance(interaction.channel, discord.TextChannel):
            parent_channel = interaction.channel
        elif isinstance(interaction.channel, discord.Thread):
            parent_channel = interaction.channel.parent

    if parent_channel is None:
        await interaction.followup.send("運用チャンネルが見つかりません。DISCORD_CHANNEL_ID を設定するか、テキストチャンネルで実行してください。", ephemeral=True)
        return

    # 親が Guild のチャンネルであることを保証
    if not isinstance(parent_channel, (discord.TextChannel, discord.ForumChannel)):
        await interaction.followup.send(
            "この場所ではスレッドを作成できません。テキストチャンネルまたはフォーラムで実行してください。",
            ephemeral=True
        )
        return

    me: discord.Member = interaction.guild.me  # Bot 自身（権限判定に使う）

    # フォーラムチャンネルは「スレッド＝投稿」なので扱いを分岐
    created_thread = None
    thread_type_used = None
    error_notes = []

    try:
        if isinstance(parent_channel, discord.ForumChannel):
            # フォーラムでは基本的に「公開スレッド」扱いになる
            if not can_make_public_thread(parent_channel, me):
                await interaction.followup.send("Bot にフォーラムでスレッドを作成する権限がありません。『チャンネルを表示』『スレッドでメッセージを送信』『公開スレッドを作成』を付与してください。", ephemeral=True)
                return
            # フォーラムでは最初のメッセージが必要
            created = await parent_channel.create_thread(
                name=f"session-{interaction.user.display_name}-{uuid.uuid4().hex[:6]}",
                content="実験スレッドを開始します。まずはワーカー名を送ってください。",
                auto_archive_duration=10080  # 7 days
            )
            created_thread = created.thread
            thread_type_used = "公開"  # Forum は公開扱い
        else:
            # まず非公開スレッドを試す（権限があれば）
            if can_make_private_thread(parent_channel, me):
                try:
                    created_thread = await parent_channel.create_thread(
                        name=f"session-{interaction.user.display_name}-{uuid.uuid4().hex[:6]}",
                        type=discord.ChannelType.private_thread,
                        invitable=False
                    )
                    # 非公開スレッドはメンバーを明示追加
                    try:
                        await created_thread.add_user(interaction.user)
                    except discord.Forbidden as e:
                        error_notes.append(f"add_user で Forbidden: {e}")
                        # 追加に失敗したら公開に切り替える
                        await created_thread.delete()
                        created_thread = None
                except discord.Forbidden as e:
                    error_notes.append(f"private_thread 作成で Forbidden: {e}")
                    created_thread = None

            # 非公開が無理なら公開スレッドへフォールバック
            if created_thread is None:
                if not can_make_public_thread(parent_channel, me):
                    await interaction.followup.send(
                        "スレッド作成権限が不足しています。必要権限: 「チャンネルを表示」「スレッドでメッセージを送信」「公開スレッドを作成」"
                        + ("、および非公開スレッド用に「非公開スレッドを作成」" if not can_make_private_thread(parent_channel, me) else ""),
                        ephemeral=True
                    )
                    # ついでに内部のエラーメモも出しておくとデバッグしやすい
                    if error_notes:
                        await interaction.followup.send("内部メモ: \n" + "\n".join(error_notes), ephemeral=True)
                    return

                created_thread = await parent_channel.create_thread(
                    name=f"session-{interaction.user.display_name}-{uuid.uuid4().hex[:6]}",
                    type=discord.ChannelType.public_thread
                )
                thread_type_used = "公開"
            else:
                thread_type_used = "非公開"
    except discord.Forbidden as e:
        await interaction.followup.send(f"スレッド作成で Forbidden が発生しました。Bot のロール権限とチャンネルのスレッド設定を確認してください。\n詳細: {e}", ephemeral=True)
        return
    except discord.HTTPException as e:
        await interaction.followup.send(f"スレッド作成時に HTTP エラーが発生しました: {e}", ephemeral=True)
        return

    if created_thread is None:
        await interaction.followup.send("スレッドの作成に失敗しました。Bot の権限をご確認ください。", ephemeral=True)
        return

    # ここからは元の処理（セッション作成→ワーカー名入力→開始）
    s = SESSIONS.create(
        user_id=interaction.user.id,
        user_display=interaction.user.display_name,
        guild_id=interaction.guild_id,
        thread_id=created_thread.id
    )

    # スレッド作成とセッション作成まで終わった直後に追加
    await interaction.followup.send(
        f"開始しました。{thread_type_used}スレッドへ移動してください → #{created_thread.name}\n"
        "スレッドでワーカー名を1行で送ると準備完了です。",
        ephemeral=True
    )

    await bot_send(
        created_thread,
        "開始前に、クラウドワークスでの、あなたの **ワーカー名（ユーザー名）** を1行で送ってください。\n"
        "お仕事完了の確認に使用するため、誤りのないように正確に記入してください。\n"
        "入力するまで実験は開始しません。キャンセルする場合は `/end` を実行してください。\n"
        "例: `username001`"
    )

    def _check(m: discord.Message) -> bool:
        return (
            (m.channel.id == created_thread.id)
            and (m.author.id == interaction.user.id)
            and (not m.author.bot)
            and (m.content.strip() != "")
        )

    name_msg: discord.Message = await bot.wait_for("message", check=_check)
    worker_name = name_msg.content.strip()
    s.conversations.append(Entry(name_msg, role="user"))

    # 確認ループ
    while True:
        view = NameConfirm(interaction.user.id, worker_name, timeout=600.0)
        prompt = (
            f"受け取ったワーカー名: **{worker_name}**\n"
            "この名前でよいですか。ボタンで選んでください。"
        )
        confirm_msg = await created_thread.send(prompt, view=view)
        await view.wait()

        if view.value is None:
            # タイムアウト
            await created_thread.send("確認がタイムアウトしました。ワーカー名をもう一度 1 行で送ってください。")
            name_msg = await bot.wait_for("message", check=_check)
            worker_name = name_msg.content.strip()
            s.conversations.append(Entry(name_msg, role="user"))
            continue

        if view.value:
            # 確定
            break
        else:
            # 修正を選んだ場合は再入力を待つ
            await created_thread.send("新しいワーカー名を 1 行で送ってください。例: `username001`")
            name_msg = await bot.wait_for("message", check=_check)
            worker_name = name_msg.content.strip()
            s.conversations.append(Entry(name_msg, role="user"))
            # ループ継続

    s.worker_name = worker_name
    s.log_file = make_log_filename(created_thread.id, worker_name)
    s.memory_file = make_memory_filename(f"thread_{created_thread.id}_{worker_name}")

    theme_msg = await bot_send(
        created_thread,
        f"{THEME_TITLE}\nこのスレッドでは上のテーマに沿って発言してください。途中で一時停止は `/pause`、再開は `/resume`。"
    )
    intro = await bot_send(
        created_thread,
        f"開始ユーザー（ワーカー名）: {worker_name}\nディスカッション開始。\n"
        "少しお待ちいただくと会話がはじまります。"
        "「返信」機能も活用しながら、ディスカッションを行ってください。"
    )

    s.conversations.append(Entry(theme_msg, "user"))
    s.conversations.append(Entry(intro, "user"))
    s.memory_streams["_global"].append({"speaker": "system", "content": f"議題開始（worker={worker_name}）"})
    save_log(s)

    loop = asyncio.get_running_loop()
    s.start(loop, run_session(s))


@bot.tree.command(name="pause", description="セッションを一時停止します", **_guild_kw())
async def pause_cmd(interaction: discord.Interaction):
    try:
        await interaction.response.defer(ephemeral=True, thinking=True)
    except Exception:
        pass
    s = SESSIONS.get_by_user(interaction.user.id)
    if not s:
        await interaction.followup.send("進行中のセッションが見つかりません。", ephemeral=True); return
    s.pause_event.clear()
    save_log(s)
    await interaction.followup.send("セッションを一時停止しました。", ephemeral=True)

@bot.tree.command(name="resume", description="セッションを再開します", **_guild_kw())
async def resume_cmd(interaction: discord.Interaction):
    try:
        await interaction.response.defer(ephemeral=True, thinking=True)
    except Exception:
        pass
    s = SESSIONS.get_by_user(interaction.user.id)
    if not s:
        await interaction.followup.send("進行中のセッションが見つかりません。", ephemeral=True); return
    s.pause_event.set()
    save_log(s)
    await interaction.followup.send("セッションを再開しました。", ephemeral=True)

@bot.tree.command(name="end", description="セッションを終了します", **_guild_kw())
async def end_cmd(interaction: discord.Interaction):
    try:
        await interaction.response.defer(ephemeral=True, thinking=True)
    except Exception:
        pass
    s = SESSIONS.get_by_user(interaction.user.id)
    if not s:
        await interaction.followup.send("進行中のセッションが見つかりません。", ephemeral=True); return
    try:
        save_log(s)
        s.stop()
    except Exception:
        pass
    SESSIONS.remove(s)
    await interaction.followup.send("セッションを終了しました。ご協力ありがとうございました。", ephemeral=True)

# ==== 会話補助 ====
def build_user_personality_update_prompt(agent_name, user_name, recent_user_msgs):
    interactions_text = "\n".join(f"- {m}" for m in recent_user_msgs)
    return (
        f"{agent_name}の視点から、最近の{user_name}の発言を踏まえて、"
        f"{user_name}の性格、関心、嗜好、特徴などについて簡潔に記憶してください。\n"
        f"ユーザの発言例:\n{interactions_text}\n"
        "具体的かつ簡潔に、二〜三文でまとめてください。"
    )

async def update_user_personality(session: Session, agent_name: str, user_name: str, recent_user_msgs: List[str]):
    prompt = build_user_personality_update_prompt(agent_name, user_name, recent_user_msgs)
    personality_summary = await asyncio.get_event_loop().run_in_executor(None, call_gemini, prompt, temp, 200)
    personality_summary = (personality_summary or "").strip()
    if personality_summary:
        session.memory_streams[agent_name]["user_memory"].append(personality_summary)

def get_user_personality_summary(session: Session, agent_name: str) -> str:
    summaries = session.memory_streams[agent_name]["user_memory"]
    return summaries[-1] if summaries else "特になし"

def build_reflection_prompt(agent_name, recent_memories):
    text = "\n".join(f"- {m}" for m in recent_memories)
    return (
        f"{agent_name}の最近の発言を抽象的かつ簡潔に二〜三文で内省してください。\n"
        f"発言一覧:\n{text}\n"
        f"{THEME_REMINDER}\n"
        f"{style_variation_hint()}\n"
    )

def get_reply_target_for_auto(conversation_list: List[Entry], current_agent_name: str) -> Optional[Entry]:
    for e in reversed(conversation_list):
        if e.role == "assistant" and e.agent and e.agent != current_agent_name:
            return e
    for e in reversed(conversation_list):
        if e.role == "user":
            return e
    return None

async def generate_agent_reply(session: Session, agent: Dict[str, Any], chain_text: str, target_length: int, term_instruction: str) -> str:
    self_history = session.memory_streams[agent["name"]]["self_memory"][-5:]
    other_histories = {}
    for other in AGENTS:
        if other["name"] == agent["name"]:
            continue
        msgs = session.memory_streams[agent["name"]]["other_memory"][other["name"]]
        other_histories[other["name"]] = msgs[-5:]

    prompt = (
        f"{build_persona(agent)}\n"
        f"{THEME_TITLE}\n"
        "【あなたのこれまでの発言】\n" + "\n".join(f"- {m}" for m in self_history) + "\n\n"
        "【他エージェントの発言】\n" + "\n".join(
            f"- {oname}: {msg}"
            for oname, msgs in other_histories.items() for msg in msgs
        ) + "\n\n"
        f"【返信対象の連続履歴】\n{chain_text}\n"
        "指示:\n"
        f"{CASUAL_STYLE}"
        f"{style_variation_hint()}"
        "この流れを踏まえた返信を作成。問いや示唆により話題を深めること。\n"
        f"{term_instruction}\n"
        "固有名詞での呼びかけは禁止。冒頭のエコー（繰り返しや反芻，相手への共感の言葉）は禁止。\n"
        f"文字数は{target_length}前後。"
    )
    reply = await asyncio.get_event_loop().run_in_executor(None, call_gemini, prompt, temp, 300)
    reply = normalize_tail_punct(compact_blank_lines(reply or ""))
    return reply

async def respond_to_user_in_session(session: Session, user_message: discord.Message, channel: discord.Thread):
    parent_entry = None
    if user_message.reference and user_message.reference.message_id:
        for e in session.conversations:
            if int(getattr(e.message, "id", 0)) == int(user_message.reference.message_id):
                parent_entry = e
                break

    if parent_entry and parent_entry.agent:
        agent = next(a for a in AGENTS if a["name"] == parent_entry.agent)
    else:
        agent = pick_agent(exclude=session.last_speaker)

    session.last_speaker = agent["name"]

    recent_user_msgs = [e.message.content for e in session.conversations[-10:] if e.role == "user"][:3]
    if len(recent_user_msgs) >= 3 and (session.turn % 10 == 0):
        await update_user_personality(session, agent["name"], user_message.author.display_name, recent_user_msgs)

    reply_target_entry = next((e for e in reversed(session.conversations)
                               if e.role == "user" and int(getattr(e.message, "id", 0)) == int(user_message.id)), None)
    reply_chain = get_reply_chain(session.conversations, reply_target_entry)
    chain_text = ""
    for e in reply_chain:
        speaker = e.agent if e.agent else (session.worker_name or "ユーザ")
        mcontent = e.message.content
        chain_text += f"{speaker}: {mcontent}\n"

    target_length = random.randint(length_short, length_long)

    term_instruction = ""
    if session.original_happiness_term.get("term") and random.random() < newterm_prob:
        term_instruction = (
            f"\n会話の要所で新語{session.original_happiness_term['term']}を自然に一回だけ用いてください。"
            "意味の説明は不要です。括弧で囲まない。"
        )

    try:
        reply = await generate_agent_reply(session, agent, chain_text, target_length, term_instruction)
    except Exception:
        reply = "もし良ければ、厚労省の基礎調査のように一週間の睡眠と食事と運動を簡単に記録すると、自分に合う幸福のパターンが見つかりやすいです。"
        reply = normalize_tail_punct(compact_blank_lines(reply))

    await asyncio.sleep(random.uniform(sleep_min, sleep_max))
    content = f"{agent['name']}: {reply}"
    sent = await bot_send(channel, content, reply_to=user_message)

    session.conversations.append(Entry(sent, role="assistant", agent=agent["name"]))
    session.turn += 1

    session.memory_streams[agent["name"]]["self_memory"].append(content)
    for other in AGENTS:
        if other["name"] != agent["name"]:
            session.memory_streams[other["name"]]["other_memory"][agent["name"]].append(content)
    session.memory_streams["_global"].append({"speaker": agent["name"], "content": reply})
    save_log(session)

async def run_session(session: Session):
    try:
        channel: discord.Thread = bot.get_channel(session.thread_id)
        if channel is None:
            return

        await asyncio.sleep(1.0)

        while True:
            await session.pause_event.wait()

            # 未処理ユーザ発言を処理
            fetch_limit = 50
            msgs = [m async for m in channel.history(limit=fetch_limit)]
            msgs = list(reversed(msgs))  # 古い順
            for m in msgs:
                if m.author.bot:
                    continue
                if session.user_id and m.author.id != session.user_id:
                    continue
                if session.last_seen_msg_id is not None and int(m.id) <= int(session.last_seen_msg_id):
                    continue
                session.conversations.append(Entry(m, role="user"))
                session.memory_streams["_global"].append({
                    "speaker": session.worker_name or m.author.display_name,  # ★ワーカー名優先
                    "content": m.content
                })
                await respond_to_user_in_session(session, m, channel)
                session.last_seen_msg_id = int(m.id)
                save_log(session)

            # 自動議論
            reply_target_entry = get_reply_target_for_auto(session.conversations, current_agent_name=session.last_speaker or "")
            chain_text = ""
            chain_entries: List[Entry] = []
            if reply_target_entry:
                chain_entries = get_reply_chain(session.conversations, reply_target_entry)
                for e in chain_entries:
                    speaker = e.agent if e.agent else (session.worker_name or "ユーザ")
                    chain_text += f"{speaker}: {e.message.content}\n"

            agent = pick_agent(exclude=session.last_speaker)
            session.last_speaker = agent["name"]

            if session.turn >= 5:
                # 直近ログ
                recent = session.memory_streams["_global"][-10:]

                # 表示用にユーザ名を正規化（ワーカー名 or Discord表示名）
                recent_display = [
                    {"speaker": map_user_speaker(session, e.get("speaker", "")), "content": e.get("content", "")}
                    for e in recent
                ]

                # 表示用配列でメモ＆新規発言テキストを生成
                recent_memo_lines = build_recent_speaker_memo(recent_display, max_speakers=4, max_chars=36)

                prev_summaries = [
                    m for m in session.memory_streams[agent["name"]]["self_memory"]
                    if str(m).startswith("[議論のまとめ]")
                ]
                prev_text = "\n".join(prev_summaries) if prev_summaries else "なし"
                new_msgs_text = "\n".join(f"- {e['speaker']}: {e['content']}" for e in recent_display)

                # ユーザ名（呼称）を明示し、必ず触れるよう指示
                user_name_for_summary = user_personal_name(session)

                integrated_prompt = (
                    f"{build_persona(agent)}\n"
                    f"{THEME_TITLE}\n"
                    "これまでのまとめ:\n" + prev_text + "\n\n"
                    "新しい発言:\n" + new_msgs_text + "\n\n"
                    f"{CASUAL_STYLE}"
                    f"{style_variation_hint()}"
                    "下にある参考メモの要点を本文に自然に織り込みつつ，"
                    "幸福とは何か，どう実現できるかについて，五〜六文で要点をまとめてください。"
                    "文頭は「ここまでの話を簡単にまとめると，」で始める。"
                    "誰がどんな発言をしていたかも織り込むこと。\n"
                    f"※{user_name_for_summary} の発言にも必ず1回以上触れること。\n"
                    "参考メモ自体（箇条書き）は出力に含めない。固有名詞での呼びかけはしない。\n"
                    "【参考メモ（出力しない）】\n"
                    + (recent_memo_lines if recent_memo_lines else "（直近メモなし）")
                )

                await asyncio.sleep(random.uniform(sleep_min, sleep_max))
                integrated = await asyncio.get_event_loop().run_in_executor(None, call_gemini, integrated_prompt, temp, 260)
                integrated = (integrated or "").strip()
                if not integrated:
                    integrated = "ここまでの話を簡単にまとめると，身近な人間関係と生活リズムと小さな挑戦の循環が幸福を底上げする点で一致してきました。"
                integrated = normalize_tail_punct(compact_blank_lines(integrated))

                tag = "[議論のまとめ] " + integrated
                session.memory_streams[agent["name"]]["self_memory"].append(tag)
                for other in AGENTS:
                    if other["name"] != agent["name"]:
                        session.memory_streams[other]["other_memory"][agent["name"]].append(tag) if isinstance(other, str) else session.memory_streams[other["name"]]["other_memory"][agent["name"]].append(tag)
                session.memory_streams["_global"].append({"speaker": agent["name"], "content": tag})

                msg = await send_via_webhook(agent_name=agent["name"], content=f"{agent['name']}: {integrated}", thread_id=session.thread_id)
                if msg:
                    session.conversations.append(Entry(msg, role="assistant", agent=agent["name"]))
                save_log(session)

                session.reflection_count += 1
                session.turn = 0
                await asyncio.sleep(random.uniform(sleep_min, sleep_max))

                # 1) 中間アンケート告知
                if session.reflection_count == mid_turn:
                    await bot_send(
                        channel,
                        "**一度ディスカッションの中間点です。下記のリンクから中間アンケートにご回答ください。**\n"
                        "回答後に `/resume` を入力して議論を再開してください。\n"
                        f"{MIDPOINT_SURVEY_URL}"
                    )
                    save_log(session)
                    if AUTO_PAUSE_AT_MID:
                        session.pause_event.clear()

                # 2) 新語提案
                if session.reflection_count == new_turn and not session.original_happiness_term.get("term"):
                    proposer = pick_agent(exclude=agent["name"])
                    prev_glob = [
                        f"{e['speaker']}: {e['content']}"
                        for e in session.memory_streams["_global"]
                        if isinstance(e.get("content"), str) and e["content"].startswith("[議論のまとめ]")
                    ]
                    summaries_text = "\n".join(prev_glob) if prev_glob else "なし"

                    coinvent_prompt = (
                        f"{build_persona(proposer)}\n"
                        f"{THEME_TITLE}\n"
                        f"これまでの議論のまとめ:\n{summaries_text}\n\n"
                        f"{CASUAL_STYLE}"
                        f"{style_variation_hint()}"
                        "議論から自然に導かれる“幸福”の新しい名前を、必ずカタカナの造語で1つだけ提案してください。"
                        "新語は必ず<>で囲い、その意味や解釈を簡潔に述べ、これまでの議論とのつながりを示してください。"
                        "会話として自然に、400字以内。固有名詞での呼びかけは禁止。"
                    )
                    proposal = await asyncio.get_event_loop().run_in_executor(None, call_gemini, coinvent_prompt, temp, 600)
                    proposal = normalize_tail_punct(compact_blank_lines((proposal or "").strip()))
                    if not proposal:
                        proposal = "新しい呼び名は＜ココロリング＞。身近な人との小さな助け合いと、自分の成長実感が輪のようにつながる状態を指します。"
                        proposal = normalize_tail_punct(proposal)

                    msg_prop = await send_via_webhook(
                        agent_name=proposer["name"],
                        content=f"{proposer['name']}: {proposal}",
                        thread_id=session.thread_id
                    )
                    if msg_prop:
                        session.conversations.append(Entry(msg_prop, role="assistant", agent=proposer["name"]))

                    # 新語抽出と記憶
                    term = None
                    m1 = re.search(r'<([^>]+)>', proposal)
                    if m1:
                        term = m1.group(1).strip()
                    else:
                        m2 = re.search(r'＜([^＞]+)＞', proposal)
                    if 'm2' in locals() and m2:
                        term = m2.group(1).strip()

                    if term:
                        desc_prompt = (
                            f"{THEME_REMINDER}\n以下の提案から新語「{term}」の意味を1〜2文で要約してください。\n---\n{proposal}\n---"
                        )
                        desc = await asyncio.get_event_loop().run_in_executor(None, call_gemini, desc_prompt, temp, 200)
                        session.original_happiness_term = {"term": term, "description": (desc or '').strip()}
                    save_log(session)

                # 3) 終了判定
                if session.reflection_count >= end_turn:
                    await bot_send(
                        channel,
                        f"**お疲れ様でした。ディスカッションは終了です。**\n最後にアンケートの回答をお願いします：\n{END_SURVEY_URL}\n\n"
                        "アンケート回答後、クラウドワークスの契約詳細画面にて「納品する」ボタンを押してください。"
                    )
                    save_log(session)
                    return

                # 4) まとめ直後の新観点
                next_agent = pick_agent(exclude=agent["name"])
                term_instruction = ""
                if session.original_happiness_term.get("term") and random.random() < newterm_prob:
                    term_instruction = f"\n会話の要所で新語{session.original_happiness_term['term']}を自然に一回だけ使用。"
                new_topic_prompt = (
                    f"{build_persona(next_agent)}\n"
                    f"{THEME_TITLE}\n"
                    f"{CASUAL_STYLE}"
                    f"{style_variation_hint()}"
                    "今の議論に新しい観点を一つだけ提案し自然に話題を広げてください。三〜四文、二百字以内。"
                    + term_instruction
                )
                new_topic_reply = await asyncio.get_event_loop().run_in_executor(None, call_gemini, new_topic_prompt, temp, 260)
                new_topic_reply = normalize_tail_punct(compact_blank_lines((new_topic_reply or "").strip()))
                msg2 = await send_via_webhook(
                    agent_name=next_agent["name"],
                    content=f"{next_agent['name']}: {new_topic_reply}",
                    thread_id=session.thread_id
                )
                if msg2:
                    session.conversations.append(Entry(msg2, role="assistant", agent=next_agent["name"]))
                save_log(session)

            else:
                target_length = random.randint(length_short, length_long)
                term_instruction = ""
                if session.original_happiness_term.get("term") and random.random() < newterm_prob:
                    term_instruction = f"\n会話の要所で新語{session.original_happiness_term['term']}を自然に一回だけ使用。"

                self_history = session.memory_streams[agent["name"]]["self_memory"][-5:]
                other_histories = {}
                for other in AGENTS:
                    if other["name"] == agent["name"]:
                        continue
                    msgs2 = session.memory_streams[agent["name"]]["other_memory"][other["name"]]
                    other_histories[other["name"]] = msgs2[-5:]

                reply_target_entry = get_reply_target_for_auto(session.conversations, current_agent_name=agent["name"])
                is_reply = False
                if reply_target_entry and random.random() < 0.6:
                    is_reply = True
                    chain_entries = get_reply_chain(session.conversations, reply_target_entry)
                    chain_text = ""
                    for e in chain_entries:
                        speaker = e.agent if e.agent else (session.worker_name or "ユーザ")
                        chain_text += f"{speaker}: {e.message.content}\n"
                    prompt = (
                        f"{build_persona(agent)}\n"
                        f"{THEME_TITLE}\n"
                        "【あなたのこれまでの発言】\n" + "\n".join(f"- {m}" for m in self_history) + "\n\n"
                        "【他エージェントの発言】\n" + "\n".join(
                            f"- {oname}: {msg}"
                            for oname, msgs in other_histories.items() for msg in msgs
                        ) + "\n\n"
                        f"【返信対象の連続履歴】\n{chain_text}\n"
                        "指示:\n"
                        f"{CASUAL_STYLE}"
                        f"{style_variation_hint()}"
                        "上記の続きとして幸福の議論を一歩深める短い返信を作る。"
                        + term_instruction + "\n"
                        f"文字数は{target_length}前後。固有名詞呼びかけ禁止。冒頭エコー禁止。"
                    )
                    reply = await asyncio.get_event_loop().run_in_executor(None, call_gemini, prompt, temp, 500)
                    reply = normalize_tail_punct(compact_blank_lines((reply or "").strip()))
                    await asyncio.sleep(random.uniform(sleep_min, sleep_max))
                    content = f"{agent['name']}: {reply}"
                    sent = await bot_send(channel, content, reply_to=reply_target_entry.message)
                    session.conversations.append(Entry(sent, role="assistant", agent=agent["name"]))
                else:
                    prompt = (
                        f"{build_persona(agent)}\n"
                        f"{THEME_TITLE}\n"
                        "【あなたのこれまでの発言】\n" + "\n".join(f"- {m}" for m in self_history) + "\n\n"
                        "【他エージェントの発言】\n" + "\n".join(
                            f"- {oname}: {msg}"
                            for oname, msgs in other_histories.items() for msg in msgs
                        ) + "\n\n"
                        "指示:\n"
                        f"{CASUAL_STYLE}"
                        f"{style_variation_hint()}"
                        "幸福の定義や測り方や実践や社会的条件などから一つだけ新しい論点を提案し自然に会話を前進させる。"
                        + term_instruction + "\n"
                        f"文字数は{target_length}前後。固有名詞呼びかけ禁止。冒頭エコー（繰り返しや反芻，相手への共感の言葉）禁止。"
                    )
                    reply = await asyncio.get_event_loop().run_in_executor(None, call_gemini, prompt, temp, 500)
                    reply = normalize_tail_punct(compact_blank_lines((reply or "").strip()))
                    await asyncio.sleep(random.uniform(sleep_min, sleep_max))
                    content = f"{agent['name']}: {reply}"
                    msg = await send_via_webhook(agent_name=agent["name"], content=content, thread_id=session.thread_id)
                    if msg:
                        session.conversations.append(Entry(msg, role="assistant", agent=agent["name"]))

                session.last_speaker = agent["name"]
                session.memory_streams[agent]["self_memory"].append(content) if isinstance(agent, str) else session.memory_streams[agent["name"]]["self_memory"].append(content)
                for other in AGENTS:
                    if other["name"] != agent["name"]:
                        session.memory_streams[other["name"]]["other_memory"][agent["name"]].append(content)
                session.memory_streams["_global"].append({"speaker": agent["name"], "content": reply})

                if len(session.memory_streams[agent["name"]]["self_memory"]) % 5 == 0:
                    recent_five = session.memory_streams[agent["name"]]["self_memory"][-5:]
                    reflection_prompt = build_reflection_prompt(agent["name"], recent_five)
                    reflection = await asyncio.get_event_loop().run_in_executor(None, call_gemini, reflection_prompt, temp, 200)
                    if reflection:
                        session.memory_streams[agent["name"]]["self_memory"].append("[内省] " + reflection)

                session.turn += 1
                save_log(session)

            await asyncio.sleep(random.uniform(sleep_min, sleep_max))

    except asyncio.CancelledError:
        try:
            save_log(session)
        except Exception:
            pass
        ch = bot.get_channel(session.thread_id)
        if ch:
            try:
                await ch.send("セッションを終了します。ご協力ありがとうございました。")
            except Exception:
                pass
        raise
    except Exception as e:
        try:
            save_log(session)
        except Exception:
            pass
        ch = bot.get_channel(session.thread_id)
        if ch:
            try:
                await ch.send(f"内部エラー: {e}")
            except Exception:
                pass

# ==== 実行 ====
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise RuntimeError("環境変数 DISCORD_BOT_TOKEN が未設定です。")
    bot.run(DISCORD_TOKEN)
