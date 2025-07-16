import sounddevice as sd
import soundfile as sf
import numpy as np
import whisper
import openai
import torchaudio
import torch
from playsound import playsound
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# 🔑 OpenAI API 키 설정 (반드시 개인 키로 변경)
openai.api_key = "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"  # 프로젝트 키 X, 개인 키 사용 권장

# 설정
DURATION = 5
SAMPLERATE = 16000
AUDIO_FILE = "user_audio.wav"
MIC_DEVICE_ID = 2

# 1️⃣ 마이크 녹음
print("🎤 녹음 시작... 말하세요.")
recording = sd.rec(int(DURATION * SAMPLERATE),
                   samplerate=SAMPLERATE,
                   channels=1,
                   dtype='float32',
                   device=MIC_DEVICE_ID)
sd.wait()

print("최대 볼륨:", np.max(np.abs(recording)))
sf.write(AUDIO_FILE, (recording * 32767).astype(np.int16), SAMPLERATE)
print("✅ 녹음 완료:", AUDIO_FILE)

# 2️⃣ Whisper 음성 인식
print("🗣️ 음성 인식 중...")
whisper_model = whisper.load_model("base")
audio_result = whisper_model.transcribe(AUDIO_FILE)
user_text = audio_result['text']
print("👉 사용자 입력:", user_text)

if user_text.strip() == "":
    print("❌ 음성 인식 실패: 입력된 음성이 없습니다.")
    exit()

# 3️⃣ GPT-4o 응답 생성
print("🤖 GPT 응답 생성 중...")
try:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "너는 프리렌처럼 조용하고 차분한 말투로 말하는 캐릭터야. 그리고 2줄 이상 넘기지 마."},
            {"role": "user", "content": user_text}
        ]
    )
    reply_text = response.choices[0].message.content.strip()
    print("💬 GPT 응답:", reply_text)

except Exception as e:
    print("❌ GPT 호출 중 오류 발생:", str(e))
    exit()

# 4️⃣ XTTS 캐릭터 음성 합성 (에러 처리 포함)
try:
    print("🛠️ XTTS 모델 로딩 중...")
    config = XttsConfig()
    config.load_json("/home/kimsj/PycharmProjects/pythonProject/TTS/run/training/GPT_XTTS_v2.0_LJSpeech_FT-May-11-2024_12+19PM-dbf1a08a/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config,
                          checkpoint_path="/home/kimsj/PycharmProjects/pythonProject/TTS/run/training/GPT_XTTS_v2.0_LJSpeech_FT-May-11-2024_12+19PM-dbf1a08a/best_model_9636.pth",
                          vocab_path="/home/kimsj/PycharmProjects/pythonProject/TTS/run/training/XTTS_v2.0_original_model_files/vocab.json",
                          use_deepspeed=True)
    model.cuda()

    print("🎙️ 화자 임베딩 생성 중...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[
        "/home/kimsj/PycharmProjects/pythonProject/240510_whisper/wavs/audio1.wav"
    ])

    print("🎧 캐릭터 음성 생성 중...")
    out = model.inference(
        reply_text,
        "ko",
        gpt_cond_latent,
        speaker_embedding,
        temperature=0.7,
    )

    torchaudio.save("output_reply.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)

except Exception as e:
    print("❌ XTTS 처리 중 오류 발생:", str(e))
    exit()

# 5️⃣ 캐릭터 음성 재생
print("🔊 캐릭터 음성 재생 중...")
try:
    playsound("output_reply.wav")
except Exception as e:
    print("❌ 음성 재생 중 오류 발생:", str(e))

print("✅ 대화 종료")