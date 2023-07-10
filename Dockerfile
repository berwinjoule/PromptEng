# 基础镜像
FROM python:3.9-slim-buster

# 设置工作目录
WORKDIR /app

# 复制应用程序代码到容器中
COPY . .

# 安装依赖项
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 18004

# 运行应用程序
CMD ["python", "api.py"]