# VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech

## For Setup

#### Step 1 (Clone the git repo)

```sh

git clone https://github.com/newton2149/vits_kinyarwanda.git

```

#### Step 2 (Install Required Dependencies) 
```sh

pip3 install -r requiremnts.txt

#Install ESpeak Engine
apt-get install espeak
```


#### Step 3 (Inference Code)
```sh
# --txt_file: "path/to/text/file"
# --device: "gpu/cpu"
# --model: "path/to/model/weights"
python3 infer.py --txt_file "./predict.txt" --device "gpu" --model "./logs/ljs_base/G_*.pth" 

```

