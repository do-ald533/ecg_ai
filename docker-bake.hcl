group "default" {
  targets = ["ecg_image"]
}

target "ecg_image" {
  context = "."
  dockerfile = "Dockerfile"
  tags = ["ecg_image:latest"]
  cache-from = ["type=local,src=.buildx-cache"]
  cache-to   = ["type=local,dest=.buildx-cache,mode=max"]
  platforms  = ["linux/amd64"]
}
