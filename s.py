import os
import shutil

# ✅ 폴더 삭제 실패 시 무시하는 안전한 삭제 함수 (사용 안함: 삭제 로직 제거)
def safe_rmtree(path):
    try:
        shutil.rmtree(path)
        print(f"✅ {path} 삭제 완료")
    except PermissionError:
        print(f"⚠️ {path} 삭제 실패: 파일이 사용 중입니다. 삭제를 건너뜁니다.")

if __name__ == "__main__":
    from trainer import Trainer, TrainerArgs
    from TTS.config.shared_configs import BaseDatasetConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
    from TTS.utils.manage import ModelManager

    # ✅ 경로 설정
    KSS_DATASET_PATH = "D:/3-GBL/archive/kss/"
    KSS_TRANSCRIPT_PATH = "D:/3-GBL/archive/kss/metadata.csv"
    OUTPUT_PATH = "D:/3-GBL/xtts_training_output/"
    CHECKPOINTS_OUT_PATH = os.path.join(OUTPUT_PATH, "XTTS_v2.0_original_model_files/")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    # ✅ 다운로드 링크
    TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"
    DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "vocab.json")
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "model.pth")
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, "dvae.pth")
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, "mel_stats.pth")

    # ✅ 사전학습 모델 다운로드
    if not all(os.path.isfile(f) for f in [TOKENIZER_FILE, XTTS_CHECKPOINT, DVAE_CHECKPOINT, MEL_NORM_FILE]):
        print("📥 모델 파일 다운로드 중...")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK, DVAE_CHECKPOINT_LINK, MEL_NORM_LINK],
            CHECKPOINTS_OUT_PATH,
            progress_bar=True
        )

    # ✅ 데이터셋 설정
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        dataset_name="kss",
        path=KSS_DATASET_PATH,
        meta_file_train=KSS_TRANSCRIPT_PATH,
        language="ko",
    )

    # ✅ 모델 설정
    model_args = GPTArgs(
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        mel_norm_file=MEL_NORM_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000
    )

    trainer_config = GPTTrainerConfig(
    output_path=OUTPUT_PATH,
    model_args=model_args,
    run_name="XTTS_KSS_finetune",
    project_name="KSS_XTTS",
    dashboard_logger="tensorboard",
    logger_uri=None,
    audio=audio_config,
    batch_size=4,
    batch_group_size=48,
    eval_batch_size=4,
    num_loader_workers=4,
    print_step=50,
    plot_step=100,
    save_step=100,             # ✅ 더 자주 저장하도록 변경
    save_n_checkpoints=5,      # ✅ 최근 5개까지 보존
    save_checkpoints=True,
    optimizer="AdamW",
    optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
    lr=5e-06,
    epochs=10,                # ✅ 추가: 원하는 epoch 수 설정
    test_sentences=[
        {
            "text": "안녕하세요. 저는 프리렌처럼 조용한 캐릭터입니다.",
            "speaker_wav": ["D:/3-GBL/archive/kss/wavs/1_0000.wav"],
            "language": "ko",
        },
    ],
)


    # ✅ 모델 및 데이터 로드
    print("📊 데이터셋 로딩 중...")
    DATASETS_CONFIG_LIST = [dataset_config]
    model = GPTTrainer.init_from_config(trainer_config)
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=256,
    )

    # ✅ 학습 시작
    print("🚀 XTTS Fine-tuning 시작...")
    trainer = Trainer(
        TrainerArgs(restore_path=None),
        trainer_config,
        output_path=OUTPUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    # ✅ 학습 완료 후 모델 강제 저장
    print("💾 학습 완료: 모델 저장 중...")
    model.save_checkpoint(os.path.join(OUTPUT_PATH, "best_model.pth"))
    print("✅ 모델 저장 완료.")
