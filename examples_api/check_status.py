import httpx

if __name__ == '__main__':
    response = httpx.get('https://prod.api.infertrade.com/status')

    print(response.status_code)