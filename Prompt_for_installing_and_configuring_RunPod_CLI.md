# Prompt for Installing and Configuring RunPod CLI and installing LingBot World diffusion model
```Prompt Okay, I need you to review the document "https://docs.runpod.io/runpodctl/overview" and "https://www.modelscope.cn/models/Robbyant/lingbot-world-v2-14b-causal-fast", I want to configure the simplest server to host that model, and you will need to do it using the runpod CLI, if you need me to authenticate, then let me know and make a way to do so, then continue working```

The target model, Robbyant/lingbot-world-v2-14b-causal-fast, a diffusion-based Image-to-Video/World Model built on Wan2.2

Run the runpod CLI command to display a summary of your currently active GPU pods

1. Generate an API Key on RunPod

2. Using the CLI, authenticate with API key and confirm the authentication:
```bash
runpodctl config --apiKey rpa_runpod_api_key
```


3. Set up a basic server to host the model. Initial configuration will involve using "torchrun --nproc_per_node=8 generate.py --task i2v-A14" as specified for rapid model initialization.

Runpod Template search for pytorch
```bash
runpodctl template search pytorch
```
Runpod GPU List
```bash
runpodctl gpu list
```

4. Perform resource allocation for hosting the model (Lingbot-World).  The documentation specifies the need for 8 GPUs, utilizing torchrun with nproc_per_node=8, to handle the 14B parameter model for 480p resolution.
```bash
runpodctl pod create --help
```
Below is LLM reasoning from Prompt above (I used Gemma 3.1 Pro) :
"Exploring GPU Requirements

I'm currently investigating the GPU requirements for Lingbot-World-v2 14B causal-fast. Their inference script suggests eight GPUs, so I'm considering options like --gpu-count 8 with NVIDIA A100s.

Selecting GPU Type

Considering the 14B model, I'm leaning towards a single 80GB GPU for simplicity and availability, specifically the NVIDIA H100 80GB HBM3 due to its high stock. This should be sufficient for inference."

5. Create the Runpod after selection:
```bash
runpodctl pod create --template-id runpod-torch-v240 --gpu-id "NVIDIA H100 80GB HBM3" --gpu-count 1 --name "lingbot-world-v2-14b-fast" --container-disk-in-gb 100 --volume-in-gb 100 --ports '8888/http,22/tcp'
```
Below is LLM reasoning from Prompt above (I used Gemma 3.1 Pro):
"I have authenticated and successfully set up the simplest server on RunPod for your model using the runpodctl CLI.

Here is a summary of the configuration and deployment:

Information review:

Checked the model card for Robbyant/lingbot-world-v2-14b-causal-fast. It requires torch >= 2.4.0, large VRAM (it's a 14B parameter diffusion model built on Wan2.2), and dependencies like flash_attn.
We used the RunPod PyTorch 2.4.0 template, as it perfectly aligns with the required dependencies and provides pre-installed CUDA 12.4.

Provided Pod Details:

Pod ID: RunPod ID here
Container Name: lingbot-world-v2-14b-fast
GPU: 1x NVIDIA H100 SXM (80GB VRAM) – This provides maximum headroom and highest availability stock on RunPod for testing out 14B VRAM-heavy inferences.
Storage: 100 GB Container Disk, 100 GB Network Volume /workspace
Ports Exposed: 8888 (Jupyter) & 22 (SSH)"

6. Launch the Model:

SSH into your new pod to prepare the inference environment:
```bash
runpodctl pod list # to check the status
runpodctl ssh runpod_id_here
```

Once inside the Pod terminal, pull the model and dependencies based on the documentation:
```bash
# Clone the repository
git clone https://github.com/robbyant/lingbot-world-v2.git
cd lingbot-world-v2

# Install dependencies and flash attention
cat requirements.txt
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install "huggingface_hub[cli]"

# Download the model
huggingface-cli download robbyant/lingbot-world-v2-14b-causal-fast --local-dir ./lingbot-world-v2-14b-causal-fast

# Example Inference Command
torchrun --nproc_per_node=1 generate.py --task i2v-A14B --size 480*832 --ckpt_dir lingbot-world-v2-14b-causal-fast --image examples/03/image.jpg --action_path examples/03 --prompt "Describe the video scene here."
```

### Just use the runpod_setup.sh script to set it up, it includes the execution of the Python script to build the API for access generate.py from LingBot-World

8. Response from Gemini 3.1 Pro on final setup:

"I have successfully established the remote API within the RunPod instance and configured a local MCP server to handle your inference workloads.

Here is the setup breakdown:

1. The RunPod Public API Endpoint
Since the Pod exposes port 8888, you can send HTTP POST requests

Created 5 todos

directly to the RunPod Proxy endpoint using your pod ID:

Endpoint: https://runpod_id_here-8888.proxy.runpod.net/generate
Method: POST
JSON Payload:
```bash
{
    "prompt": "Your description of the scene.",
    "image_path": "examples/03/image.jpg",
    "action_path": "examples/03", 
    "frame_num": 361
}
```

(Note: The server is currently downloading the 86GB model and setting up flash_attn in the background. It will take a few minutes before the endpoint starts accepting requests.)

2. Local MCP Server
I created an MCP Server Python script in your workspace at local_workspace/lingbot_mcp_server.py. That script declares an MCP tool called generate_lingbot_video which wraps the REST call cleanly to your remote RunPod instance.

Requirements for your local machine:
You need mcp and httpx installed locally.
```bash
pip install "mcp[cli]" httpx
```

3. How to Execute via MCP
You can integrate this tool into any standard MCP Client (like Cursor, Claude Desktop, or your VS Code extensions) using the config file I generated (local_workspace/lingbot_mcp_config.json):
```bash
{
  "mcpServers": {
    "lingbot-world": {
      "command": "python",
      "args": ["/local_workspace/lingbot_mcp_server.py"]
    }
  }
}
```

Monitoring the Setup
If you would like to track the progress of the remote model downloading and initialization, you can inspect the background logs directly from your local terminal:
```bash
ssh -i /root/.runpod/ssh/RunPod-Key-Go -p 19501 root@public_ip_address "tail -f /workspace/setup.log"
```
Once the setup.log prints "Server started.", the MCP Tool requests will seamlessly route straight to the PyTorch RunPod inference runtime!
"