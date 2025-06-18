import requests

# ✅ 디스코드 웹훅 주소
WEBHOOK_URL = "https://discord.com/api/webhooks/1384916960889929838/Fg3FQMRt7fRhZ5kvSDT4AsRxRzZFyf6_kZ8T6Oz2nrvY6uk0eGtW4uFvw1LT4VmDnSYm"

# ngrok에서 현재 활성화된 https 주소 가져오기
def get_ngrok_url():
    try:
        res = requests.get("http://127.0.0.1:4040/api/tunnels")
        tunnels = res.json()["tunnels"]
        for tunnel in tunnels:
            if tunnel["proto"] == "https":
                return tunnel["public_url"]
    except Exception as e:
        print("ngrok 주소를 가져오지 못했습니다:", e)
        return None

# 디스코드로 메시지 전송
def send_to_discord(message):
    data = {"content": message}
    response = requests.post(WEBHOOK_URL, json=data)
    if response.status_code == 204:
        print("✅ 디스코드 전송 완료!")
    else:
        print("❌ 전송 실패:", response.text)

# 메인 실행
url = get_ngrok_url()
if url:
    send_to_discord(f"🟢 새로운 ngrok 주소가 생성되었습니다:\n{url}")
else:
    print("❌ ngrok가 실행 중인지 확인하세요.")
