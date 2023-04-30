import requests
import os

if __name__ == '__main__':
    proxy = "http://192.168.126.12:12798"
    proxies = {
        "http": proxy,
        "https": proxy,
    }

    url = "https://civitai.com/api/download/models/46846"

    # use proxy
    # os.environ['HTTP_PROXY'] = 'http://192.168.126.12:12798'
    # os.environ['HTTPS_PROXY'] = 'http://192.168.126.12:12798'
    print(url)

    model_ckpt_dir = '/root/autodl-tmp/models/ui_models_1'
    with open(f'{model_ckpt_dir}/rev_animated/v1.2.2.SafeTensor', 'wb') as f:
        # 允许重定向, 允许分段下载, 允许使用代理加速
        res = requests.get(url, proxies=proxies, allow_redirects=True, stream=True)
        for chunk in res.iter_content(chunk_size=10240):
            f.write(chunk)
