# transcript 파일 경로
transcript_path = 'D:/3-GBL/archive/transcript.v.1.4.txt'
fixed_transcript_path = 'D:/3-GBL/archive/transcript_fixed.txt'

with open(transcript_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed_lines = []
for line in lines:
    if line.startswith('wavs/') or line.startswith('wavs\\'):
        # wavs/ 또는 wavs\ 삭제
        line = line.replace('wavs/', '').replace('wavs\\', '')
    fixed_lines.append(line)

# 수정된 파일 저장
with open(fixed_transcript_path, 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("수정된 transcript 파일이 저장되었습니다.")
