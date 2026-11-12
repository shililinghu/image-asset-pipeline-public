import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
BACKUP_DIR = BASE_DIR / "backups"
MOCK_IMAGE_OUTPUT_DIR = BASE_DIR / "exports" / "mock_images"
REAL_IMAGE_OUTPUT_DIR = BASE_DIR / "exports" / "real_images"
IMAGE_OUTPUT_DIR = MOCK_IMAGE_OUTPUT_DIR
INPUT_IMAGES_DIR = BASE_DIR / "input_images"
LLM_ERROR_LOG = LOG_DIR / "llm_errors.log"

IDEA_REQUESTS_FILE = DATA_DIR / "idea_requests.json"
SUBJECT_PROFILES_FILE = DATA_DIR / "subject_profiles.json"
PROMPT_PACKS_FILE = DATA_DIR / "prompt_packs.json"
PROMPT_PACK_ITEMS_FILE = DATA_DIR / "prompt_pack_items.json"
PROMPT_PACK_MATCH_LOGS_FILE = DATA_DIR / "prompt_pack_match_logs.json"
PROMPT_PACK_RUNS_FILE = DATA_DIR / "prompt_pack_runs.json"
PROMPT_PACK_BUILD_RUNS_FILE = DATA_DIR / "prompt_pack_build_runs.json"
IMAGE_GENERATION_TASKS_FILE = DATA_DIR / "image_generation_tasks.json"
IMAGE_RESULTS_FILE = DATA_DIR / "image_results.json"
IMAGE_REVIEWS_FILE = DATA_DIR / "image_reviews.json"
HUMAN_IMAGE_REVIEWS_FILE = DATA_DIR / "human_image_reviews.json"
PROMPT_IMAGE_FEEDBACK_FILE = DATA_DIR / "prompt_image_feedback.json"
IMAGE_PROMPT_EXTRACTIONS_FILE = DATA_DIR / "image_prompt_extractions.json"
CANDIDATE_PROMPT_RUNS_FILE = DATA_DIR / "candidate_prompt_runs.json"
PROMPT_PACK_INCUBATION_RUNS_FILE = DATA_DIR / "prompt_pack_incubation_runs.json"
PROMPT_VERSIONS_FILE = DATA_DIR / "prompt_versions.json"
PROMPT_EVOLUTION_RUNS_FILE = DATA_DIR / "prompt_evolution_runs.json"
PROMPT_VERSION_ARCHIVES_FILE = DATA_DIR / "prompt_version_archives.json"
APP_SETTINGS_FILE = DATA_DIR / "app_settings.json"
SECRETS_LOCAL_FILE = DATA_DIR / "secrets.local.json"

DATA_FILES = {
    "projects": DATA_DIR / "projects.json",
    "prompts": DATA_DIR / "prompts.json",
    "test_cases": DATA_DIR / "test_cases.json",
    "test_results": DATA_DIR / "test_results.json",
    "scores": DATA_DIR / "scores.json",
    "feedback": DATA_DIR / "feedback.json",
    "iterations": DATA_DIR / "iterations.json",
    "idea_requests": IDEA_REQUESTS_FILE,
    "subject_profiles": SUBJECT_PROFILES_FILE,
    "prompt_pack_match_logs": PROMPT_PACK_MATCH_LOGS_FILE,
    "prompt_packs": PROMPT_PACKS_FILE,
    "prompt_pack_items": PROMPT_PACK_ITEMS_FILE,
    "prompt_pack_runs": PROMPT_PACK_RUNS_FILE,
    "prompt_pack_build_runs": PROMPT_PACK_BUILD_RUNS_FILE,
    "image_generation_tasks": IMAGE_GENERATION_TASKS_FILE,
    "image_results": IMAGE_RESULTS_FILE,
    "image_reviews": IMAGE_REVIEWS_FILE,
    "human_image_reviews": HUMAN_IMAGE_REVIEWS_FILE,
    "prompt_image_feedback": PROMPT_IMAGE_FEEDBACK_FILE,
    "image_prompt_extractions": IMAGE_PROMPT_EXTRACTIONS_FILE,
    "candidate_prompt_runs": CANDIDATE_PROMPT_RUNS_FILE,
    "prompt_pack_incubation_runs": PROMPT_PACK_INCUBATION_RUNS_FILE,
    "prompt_versions": PROMPT_VERSIONS_FILE,
    "prompt_evolution_runs": PROMPT_EVOLUTION_RUNS_FILE,
    "prompt_version_archives": PROMPT_VERSION_ARCHIVES_FILE,
}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek").strip().lower() or "deepseek"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip() or "https://api.deepseek.com"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash").strip() or "deepseek-v4-flash"
DEEPSEEK_SUPPORTED_MODELS = [
    "deepseek-v4-flash",
    "deepseek-v4-pro",
]
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60").strip() or "60")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7").strip() or "0.7")
MANUAL_MODE = not bool(DEEPSEEK_API_KEY)

IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "mock").strip().lower() or "mock"
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "mock-image-model").strip() or "mock-image-model"
IMAGE_API_KEY_ENV = os.getenv("IMAGE_API_KEY_ENV", "IMAGE_API_KEY").strip() or "IMAGE_API_KEY"
IMAGE_API_BASE_URL = os.getenv("IMAGE_API_BASE_URL", "").strip()
IMAGE_REQUEST_TIMEOUT = float(os.getenv("IMAGE_REQUEST_TIMEOUT", "60").strip() or "60")
IMAGE_MAX_RETRIES = int(os.getenv("IMAGE_MAX_RETRIES", "2").strip() or "2")
IMAGE_DOWNLOAD_REMOTE = os.getenv("IMAGE_DOWNLOAD_REMOTE", "true").strip().lower() not in {"0", "false", "no"}
IMAGE_HTTP_METHOD = os.getenv("IMAGE_HTTP_METHOD", "POST").strip().upper() or "POST"
IMAGE_HTTP_AUTH_TYPE = os.getenv("IMAGE_HTTP_AUTH_TYPE", "bearer").strip().lower() or "bearer"
IMAGE_HTTP_RESPONSE_IMAGE_URL_PATH = os.getenv("IMAGE_HTTP_RESPONSE_IMAGE_URL_PATH", "data.0.url").strip() or "data.0.url"
IMAGE_HTTP_RESPONSE_BASE64_PATH = os.getenv("IMAGE_HTTP_RESPONSE_BASE64_PATH", "data.0.b64_json").strip() or "data.0.b64_json"
IMAGE_HTTP_PAYLOAD_STYLE = os.getenv("IMAGE_HTTP_PAYLOAD_STYLE", "openai_compatible").strip().lower() or "openai_compatible"
IMAGE_DEFAULT_SIZE = os.getenv("IMAGE_DEFAULT_SIZE", "1024x1024").strip() or "1024x1024"
IMAGE_PROVIDER_PROFILE_PATH = os.getenv("IMAGE_PROVIDER_PROFILE_PATH", "").strip()
ENABLE_PROMPT_VARIATION_NONCE = os.getenv("ENABLE_PROMPT_VARIATION_NONCE", "true").strip().lower() not in {"0", "false", "no"}
PROMPT_VARIATION_STYLE = os.getenv("PROMPT_VARIATION_STYLE", "soft_instruction").strip() or "soft_instruction"
ENABLE_IMAGE_RESULT_CACHE = os.getenv("ENABLE_IMAGE_RESULT_CACHE", "false").strip().lower() in {"1", "true", "yes"}

AESTHETIC_REVIEW_PROVIDER = os.getenv("AESTHETIC_REVIEW_PROVIDER", "mock").strip().lower() or "mock"
AESTHETIC_REVIEW_API_URL = os.getenv("AESTHETIC_REVIEW_API_URL", "").strip()
AESTHETIC_REVIEW_API_KEY_ENV = os.getenv("AESTHETIC_REVIEW_API_KEY_ENV", "TASTELENS_API_KEY").strip() or "TASTELENS_API_KEY"
AESTHETIC_REVIEW_TIMEOUT = float(os.getenv("AESTHETIC_REVIEW_TIMEOUT", "60").strip() or "60")
AESTHETIC_REVIEW_IMAGE_FIELD = os.getenv("AESTHETIC_REVIEW_IMAGE_FIELD", "image").strip() or "image"
AESTHETIC_REVIEW_REQUEST_STYLE = os.getenv("AESTHETIC_REVIEW_REQUEST_STYLE", "multipart").strip().lower() or "multipart"
AESTHETIC_REVIEW_SCORE_FIELD = os.getenv("AESTHETIC_REVIEW_SCORE_FIELD", "displayScore").strip() or "displayScore"
AESTHETIC_REVIEW_RATING_FIELD = os.getenv("AESTHETIC_REVIEW_RATING_FIELD", "rating").strip() or "rating"
AESTHETIC_REVIEW_REASON_FIELD = os.getenv("AESTHETIC_REVIEW_REASON_FIELD", "analysis").strip() or "analysis"

WUYIN_API_KEY_ENV = "WUYIN_API_KEY"
WUYIN_BASE_URL = os.getenv("WUYIN_BASE_URL", "https://api.wuyinkeji.com/api/async").strip() or "https://api.wuyinkeji.com/api/async"
WUYIN_SUBMIT_PATH = os.getenv("WUYIN_SUBMIT_PATH", "/image_gpt").strip() or "/image_gpt"
WUYIN_DETAIL_PATH = os.getenv("WUYIN_DETAIL_PATH", "/detail").strip() or "/detail"
WUYIN_SUBMIT_TASK_ID_PATH = os.getenv("WUYIN_SUBMIT_TASK_ID_PATH", "data.id").strip() or "data.id"
WUYIN_DETAIL_SUCCESS_CODES = [1, 200]
WUYIN_DETAIL_PENDING_CODES = [0]
WUYIN_DETAIL_FAILED_CODES = [2]
WUYIN_DETAIL_STATE_PATH = os.getenv("WUYIN_DETAIL_STATE_PATH", "data.status").strip() or "data.status"
WUYIN_DETAIL_RESULT_PATH = os.getenv("WUYIN_DETAIL_RESULT_PATH", "data.result").strip() or "data.result"
WUYIN_TASK_SUCCESS_STATES = [
    "success",
    "succeeded",
    "completed",
    "complete",
    "done",
    "finished",
    "SUCCESS",
    "SUCCEEDED",
    "COMPLETED",
    1,
    2,
    200,
]
WUYIN_TASK_PENDING_STATES = [
    "pending",
    "processing",
    "running",
    "waiting",
    "queued",
    "submitted",
    "generating",
    "PROCESSING",
    "RUNNING",
    "PENDING",
    0,
]
WUYIN_TASK_FAILED_STATES = [
    "failed",
    "fail",
    "error",
    "rejected",
    "canceled",
    "cancelled",
    "FAILED",
    "ERROR",
]
WUYIN_DETAIL_IMAGE_URL_PATH = os.getenv("WUYIN_DETAIL_IMAGE_URL_PATH", "data.result").strip() or "data.result"
WUYIN_POLL_INTERVAL = float(os.getenv("WUYIN_POLL_INTERVAL", "2").strip() or "2")
WUYIN_MAX_POLL_SECONDS = float(os.getenv("WUYIN_MAX_POLL_SECONDS", "180").strip() or "180")
WUYIN_DOWNLOAD_REMOTE = os.getenv("WUYIN_DOWNLOAD_REMOTE", "true").strip().lower() not in {"0", "false", "no"}

PACK_MATCH_THRESHOLD = 75
PACK_WEAK_MATCH_THRESHOLD = 50

PROMPT_PACK_MIN_PROMPTS = 3
PROMPT_MIN_TESTED_IMAGES = 3
PROMPT_MIN_AVG_AUTO_SCORE = 75
PROMPT_MIN_HUMAN_SCORE = 4
PACK_MIN_AVG_SCORE = 75
PACK_MIN_KEEP_RATE = 0.3
PACK_MAX_SERIOUS_FAIL_RATE = 0.2
INCUBATION_AUTO_PASS_SCORE = 70
HUMAN_REVIEW_REQUIRED_FOR_PACK = True
SETTINGS_MODE = os.getenv("SETTINGS_MODE", "local").strip().lower() or "local"

DEFAULT_EXPECTED_FOCUS = "\n".join(
    [
        "贴合需求",
        "清晰可执行",
        "主体突出",
        "风格自然",
        "避免空泛和跑题",
    ]
)

GENERAL_SCORE_DIMENSIONS = [
    "贴合需求",
    "结构清晰",
    "表达吸引力",
    "可执行性",
    "用户风格匹配度",
    "可复用性",
]
