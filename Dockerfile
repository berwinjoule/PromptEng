# 基础镜像
FROM python:3.9-slim-buster

# 设置工作目录
WORKDIR /app

# 复制应用程序代码到容器中
COPY . .

# 安装依赖项
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y git

RUN git clone https://huggingface.co/alibaba-pai/pai-bloom-1b1-text2prompt-sd

# 暴露端口
EXPOSE 7861

# 运行应用程序
CMD ["python", "api.py"]