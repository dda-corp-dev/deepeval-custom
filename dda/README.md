# DeepEval DDA

## 목적

Red teaming 수행을 위해 DeepEval 프레임워크를 익히고, 자유롭게 테스트하며 DDA에 맞게 커스텀한다.

<br/>

## 주의사항

deepeval을 fork한 레포지토리이기 때문에 dda 폴더를 제외한 나머지 폴더들은 업데이트가 자주 발생합니다.

<br/>

## 설치 방법

프로젝트는 Poetry 환경에서 실행됩니다. 설치 방법은 Mac OS(m1)을 기준으로 작성되었습니다.

### poetry 설치

```bash
brew install pipx
pipx ensurepath
```

### 필요한 패키지 설치

poetry.lock 파일 삭제 후, poetry를 install하여 필요한 패키지를 설치하세요.

```bash
poetry install
```

### 인터프리터 변경

설치한 poetry 인터프리터로 변경하세요. (VSCODE 기준, F11 키 입력 후 "Python: Select Interpreter" 선택)

### 환경변수 설정

.deepeval 파일을 생성한 후, 필요한 환경변수를 설정하세요. <br/>
등록한 환경변수를 사용하려면 deepeval/key_handler.py 파일의 KeyValues 클래스에도 선언해야 합니다.
