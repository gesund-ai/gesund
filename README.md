# Gesund SDK

The Gesund SDK is an open source library for interacting with the Gesund.ai platform.

## Installation

You can install the library using pip:

```bash
pip install gesund
```

## Usage

Here's an example of how to use the Gesund SDK:

```python
from gesund import GesundSDK

sdk = GesundSDK(api_key="your_api_key")
sdk.connect()
data = sdk.get_data()
print(data)
```
