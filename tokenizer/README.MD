# Tokenizer

The `tokenizer` package provides a high-level Go interface for encoding and decoding text using tokenizer models backed by a Rust library. It abstracts the complexities of interacting with the underlying C-based tokenizer library, offering a seamless and efficient way to process text for natural language processing (NLP) tasks.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
    - [Creating a Tokenizer](#creating-a-tokenizer)
    - [Encoding Text](#encoding-text)
    - [Decoding Token IDs](#decoding-token-ids)
- [Options](#options)
    - [Encode Options](#encode-options)
    - [Decode Options](#decode-options)
- [Response Models](#response-models)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Features

- **High-Level API:** Simplifies interactions with the underlying tokenizer library.
- **Encoding & Decoding:** Convert text to token IDs and vice versa with customizable options.
- **Cross-Platform Support:** Compatible with macOS (darwin) and Linux on both amd64 and arm64 architectures.
- **Flexible Configuration:** Supports various encoding and decoding options to cater to different NLP requirements.

## Installation

### Prerequisites

- **Go:** Ensure you have Go installed (version 1.16 or later recommended). You can download it from [golang.org](https://golang.org/dl/).
- **C Compiler:** A C compiler is required to build the C-based tokenizer library. On macOS, you can use `clang`, and on Linux, `gcc` is commonly used.
- **Rust:** The tokenizer library is built in Rust. Install Rust by following the instructions at [rust-lang.org](https://www.rust-lang.org/tools/install).

### Install the Package

To install the `tokenizer` package, run:

```sh
go get github.com/Trendyol/go-triton-client/tokenizer
```

## Usage

The tokenizer package provides a simple interface to encode and decode text using tokenizer models. Below are the primary functionalities:

### Creating Tokenizer

To create a new tokenizer instance, use the NewTokenizer function by providing the path to your tokenizer model file.

```go
package main

import (
    "fmt"
    "log"
	
    "github.com/Trendyol/go-triton-client/tokenizer"
)

func main() {
    // Initialize the tokenizer with the path to the model file
    tk := tokenizer.NewTokenizer("path/to/tokenizer.json")
    defer func() {
        if err := tk.Close(); err != nil {
            log.Fatalf("Failed to close tokenizer: %v", err)
        }
    }()

    fmt.Println("Tokenizer initialized successfully.")
}
```

### Encoding Text

Encode a string into token IDs and other attributes using the `Encode` method.

```go
package main

import (
    "fmt"
    "log"

    "github.com/Trendyol/go-triton-client/tokenizer"
    "github.com/Trendyol/go-triton-client/tokenizer/options"
)

func main() {
    tk := tokenizer.NewTokenizer("path/to/tokenizer.json")
    defer tk.Close()

    text := "Hello, world!"
    encodeOptions := &options.EncodeOptions{
        ReturnAllAttributes:     true,
        ReturnAttentionMask:     true,
        ReturnTokens:            true,
        ReturnOffsets:           true,
        ReturnSpecialTokensMask: true,
        ReturnTypeIDs:           true,
        EncodeSpecialTokens:     true,
    }

    encodeResponse := tk.Encode(text, encodeOptions)

    fmt.Printf("Encoded IDs: %v\n", encodeResponse.IDs)
    fmt.Printf("Tokens: %v\n", encodeResponse.Tokens)
    fmt.Printf("Offsets: %v\n", encodeResponse.Offsets)
    // ... print other attributes as needed
}
```

### Decoding Token IDs

Decode a slice of token IDs back into a string using the `Decode` method.

```go
package main

import (
    "fmt"
    "log"

    "github.com/Trendyol/go-triton-client/tokenizer"
    "github.com/Trendyol/go-triton-client/tokenizer/options"
)

func main() {
    tk := tokenizer.NewTokenizer("path/to/tokenizer.json")
    defer tk.Close()

    tokenIDs := []uint32{101, 102, 103}
    decodeOptions := &options.DecodeOptions{
        SkipSpecialTokens: true,
    }

    decodeResponse := tk.Decode(tokenIDs, decodeOptions)

    fmt.Printf("Decoded Text: %s\n", decodeResponse.Decoded)
}
```

## Options

The `tokenizer` package allows you to customize the encoding and decoding processes using options. These options enable you to specify which attributes to return during encoding and how to handle special tokens during decoding.

### Encode Options

Defined in the `github.com/Trendyol/go-triton-client/tokenizer/options` package, `EncodeOptions` allow you to specify which attributes to include in the encoding response.

```go
type EncodeOptions struct {
    ReturnAllAttributes     bool
    ReturnAttentionMask     bool
    ReturnTokens            bool
    ReturnOffsets           bool
    ReturnSpecialTokensMask bool
    ReturnTypeIDs           bool
    EncodeSpecialTokens     bool
}
```

 - ReturnAllAttributes: If `true`, returns all available attributes.
 - ReturnAttentionMask: Includes the attention mask in the response.
 - ReturnTokens: Includes the token strings in the response.
 - ReturnOffsets: Includes the character offsets of tokens.
 - ReturnSpecialTokensMask: Includes a mask indicating special tokens.
 - ReturnTypeIDs: Includes type IDs for tokens.
 - EncodeSpecialTokens: If `true`, encodes special tokens.

### Decode Options

Defined in the `github.com/Trendyol/go-triton-client/tokenizer/options` package, `DecodeOptions` allow you to specify how to handle special tokens during decoding.

```go
type DecodeOptions struct {
    SkipSpecialTokens bool
}
```

 - SkipSpecialTokens: If `true`, skips special tokens during decoding.

## Response Models

The tokenizer package returns responses encapsulated in the models package.

### Encode Response

Defined in `github.com/Trendyol/go-triton-client/tokenizer/models`, `EncodeResponse` contains the results of the encoding process.

```go
type EncodeResponse struct {
    IDs               []uint32
    TypeIDs           []uint32
    SpecialTokensMask []uint32
    AttentionMask     []uint32
    Tokens            []string
    Offsets           []Offset
}
```

 - IDs: Slice of token IDs.
 - TypeIDs: Slice of type IDs corresponding to tokens.
 - SpecialTokensMask: Mask indicating which tokens are special tokens.
 - AttentionMask: Attention mask indicating which tokens should be attended to.
 - Tokens: Slice of token strings.
 - Offsets: Slice of `Offset` structs indicating the start and end character positions of each token.

### Decode Response

Defined in `github.com/Trendyol/go-triton-client/tokenizer/models`, `DecodeResponse` contains the result of the decoding process.

```go
type DecodeResponse struct {
    Decoded string
}
```

 - Decoded: The decoded string from token IDs.

## Examples

Below are some practical examples demonstrating how to use the tokenizer package for encoding and decoding text.

### Example: Basic Encoding and Decoding

```go
package main

import (
    "fmt"
    "log"

    "github.com/Trendyol/go-triton-client/tokenizer"
    "github.com/Trendyol/go-triton-client/tokenizer/models"
    "github.com/Trendyol/go-triton-client/tokenizer/options"
)

func main() {
    // Initialize the tokenizer
    tk := tokenizer.NewTokenizer("path/to/tokenizer.json")
    defer tk.Close()

    // Text to encode
    text := "Hello, world!"

    // Encoding options
    encodeOptions := &options.EncodeOptions{
        ReturnAllAttributes:     true,
        ReturnTokens:            true,
        EncodeSpecialTokens:     true,
    }

    // Encode the text
    encodeResp := tk.Encode(text, encodeOptions)

    fmt.Println("Encoded IDs:", encodeResp.IDs)
    fmt.Println("Tokens:", encodeResp.Tokens)
    fmt.Println("Offsets:", encodeResp.Offsets)

    // Decode the token IDs back to text
    decodeOptions := &options.DecodeOptions{
        SkipSpecialTokens: true,
    }

    decodeResp := tk.Decode(encodeResp.IDs, decodeOptions)

    fmt.Println("Decoded Text:", decodeResp.Decoded)
}
```

### Example: Encoding with Default Options

```go
package main

import (
    "fmt"
    "log"

    "github.com/Trendyol/go-triton-client/tokenizer"
)

func main() {
    // Initialize the tokenizer without specifying options
    tk := tokenizer.NewTokenizer("path/to/tokenizer.json")
    defer tk.Close()

    text := "Sample text for encoding."

    // Encode with default options (encodeOptions is nil)
    encodeResp := tk.Encode(text, nil)

    fmt.Println("Encoded IDs:", encodeResp.IDs)
    fmt.Println("Tokens:", encodeResp.Tokens)
}
```

### Example: Decoding Without Skipping Special Tokens

```go
package main

import (
    "fmt"
    "log"

    "github.com/Trendyol/go-triton-client/tokenizer"
    "github.com/Trendyol/go-triton-client/tokenizer/models"
    "github.com/Trendyol/go-triton-client/tokenizer/options"
)

func main() {
    tk := tokenizer.NewTokenizer("path/to/tokenizer.json")
    defer tk.Close()

    tokenIDs := []uint32{101, 202, 303, 404}

    // Decode without skipping special tokens
    decodeOptions := &options.DecodeOptions{
        SkipSpecialTokens: false,
    }

    decodeResp := tk.Decode(tokenIDs, decodeOptions)

    fmt.Println("Decoded Text:", decodeResp.Decoded)
}
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

### Steps to Contribute

#### 1. Fork The Repository

Click the "Fork" button at the top right of the repository page to create a copy in your GitHub account.

#### 2. Clone Your Fork

```
git clone https://github.com/your-username/go-triton-client.git
cd go-triton-client
```

#### 3. Create a Feature Branch

```
git checkout -b feature/your-feature-name
```

#### 4. Make Your Changes

Implement your feature or fix the bug.

#### 5. Run Tests

Ensure all tests pass.
```
go test ./...
```

#### 6. Commit Your Changes

```
git add .
git commit -m "Add feature: your feature description"
```

#### 7. Push to Your Fork

```
git push origin feature/your-feature-name
```

#### 8. Create a Pull Request


