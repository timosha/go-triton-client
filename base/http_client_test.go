package base

import (
	"bytes"
	"crypto/tls"
	"errors"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestNewHttpClient(t *testing.T) {
	client := NewHttpClient(5000, true, nil)
	if client == nil {
		t.Fatal("Expected non-nil HttpClient")
	}

	customClient := &http.Client{}
	client = NewHttpClient(5000, true, customClient)
	if client == nil {
		t.Fatal("Expected non-nil HttpClient with custom client")
	}
}

func TestHttpClient_Get(t *testing.T) {
	handler := func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("Expected GET method, got %s", r.Method)
		}
		if r.URL.Path != "/test" {
			t.Errorf("Expected path /test, got %s", r.URL.Path)
		}
		if r.Header.Get("Header-Key") != "Header-Value" {
			t.Errorf("Expected header Header-Key to be Header-Value")
		}
		if r.URL.Query().Get("param") != "value" {
			t.Errorf("Expected query param param to be value")
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	}
	server := httptest.NewServer(http.HandlerFunc(handler))
	defer server.Close()

	client := NewHttpClient(5000, true, nil)
	headers := map[string]string{"Header-Key": "Header-Value"}
	queryParams := map[string]string{"param": "value"}
	resp, err := client.Get(server.URL, "test", headers, queryParams)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status 200, got %d", resp.StatusCode)
	}
}

func TestHttpClient_Do(t *testing.T) {
	client := NewHttpClient(5000, true, nil)
	req, _ := http.NewRequest(http.MethodGet, "unreachable://example.com", nil)
	_, err := client.Do(req)
	if err == nil {
		t.Errorf("Expected error for unreachable host")
	}
}

func TestHttpClient_AddHeaders(t *testing.T) {
	client := &httpClient{}
	req, _ := http.NewRequest(http.MethodGet, "http://example.com", nil)
	headers := map[string]string{"Header-Key": "Header-Value"}
	client.addHeaders(req, headers)
	if req.Header.Get("Header-Key") != "Header-Value" {
		t.Errorf("Expected header Header-Key to be Header-Value")
	}
}

func TestHttpClient_AddQueryParameters(t *testing.T) {
	client := &httpClient{}
	req, _ := http.NewRequest(http.MethodGet, "http://example.com", nil)
	queryParams := map[string]string{"param": "value"}
	client.addQueryParameters(req, queryParams)
	if req.URL.Query().Get("param") != "value" {
		t.Errorf("Expected query param param to be value")
	}
}

func TestHttpClient_Get_Error(t *testing.T) {
	client := NewHttpClient(5000, true, nil)
	_, err := client.Get(":", "test", nil, nil)
	if err == nil {
		t.Errorf("Expected error due to invalid URL")
	}
}

func TestHttpClient_Post_Error(t *testing.T) {
	client := NewHttpClient(5000, true, nil)
	_, err := client.Post(":", "test", "body", nil, nil)
	if err == nil {
		t.Errorf("Expected error due to invalid URL")
	}
}

func TestHttpClient_PostWithBytes_Error(t *testing.T) {
	client := NewHttpClient(5000, true, nil)
	_, err := client.PostWithBytes(":", "test", []byte("body"), nil, nil)
	if err == nil {
		t.Errorf("Expected error due to invalid URL")
	}
}

func TestNewHttpClient_CustomClient(t *testing.T) {
	customTransport := &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	}
	customClient := &http.Client{
		Timeout:   10 * time.Second,
		Transport: customTransport,
	}
	client := NewHttpClient(0, false, customClient)
	httpClient, ok := client.(*httpClient)
	if !ok {
		t.Fatalf("Expected *httpClient type")
	}
	if httpClient.client != customClient {
		t.Errorf("Expected custom client to be used")
	}
}

func TestNewHttpClient_DefaultClient(t *testing.T) {
	client := NewHttpClient(5000, true, nil)
	httpClient, ok := client.(*httpClient)
	if !ok {
		t.Fatalf("Expected *httpClient type")
	}
	transport, ok := httpClient.client.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("Expected *http.Transport type")
	}
	if transport.TLSClientConfig.InsecureSkipVerify != true {
		t.Errorf("Expected InsecureSkipVerify to be true")
	}
	if httpClient.client.Timeout != 5000*time.Millisecond {
		t.Errorf("Expected timeout to be 5000ms")
	}
}

func TestHttpClient_Do_Error(t *testing.T) {
	transportErr := errors.New("transport error")
	expectedErr := "Get \"http://example.com\": transport error"
	client := &httpClient{
		client: &http.Client{
			Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				return nil, transportErr
			}),
		},
	}
	req, err := http.NewRequest(http.MethodGet, "http://example.com", nil)
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}
	_, err = client.Do(req)
	if err == nil || err.Error() != expectedErr {
		t.Errorf("Expected transport error, got %v", err)
	}
}

func TestHttpClient_Get_DoError(t *testing.T) {
	client := &httpClient{
		client: &http.Client{
			Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				return nil, errors.New("do error")
			}),
		},
	}
	expectedErr := "Get \"http://example.com/test\": do error"
	_, err := client.Get("http://example.com", "test", nil, nil)
	if err == nil || err.Error() != expectedErr {
		t.Errorf("Expected 'do error', got %v", err)
	}
}

func TestHttpClient_Post(t *testing.T) {
	handler := func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("Expected POST method, got %s", r.Method)
		}
		if r.URL.Path != "/test" {
			t.Errorf("Expected path /test, got %s", r.URL.Path)
		}
		body, _ := ioutil.ReadAll(r.Body)
		if string(body) != "request body" {
			t.Errorf("Expected body to be 'request body', got '%s'", string(body))
		}
		if r.Header.Get("Header-Key") != "Header-Value" {
			t.Errorf("Expected header Header-Key to be Header-Value")
		}
		if r.URL.Query().Get("param") != "value" {
			t.Errorf("Expected query param param to be value")
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	}
	server := httptest.NewServer(http.HandlerFunc(handler))
	defer server.Close()

	client := NewHttpClient(5000, true, nil)
	headers := map[string]string{"Header-Key": "Header-Value"}
	queryParams := map[string]string{"param": "value"}
	resp, err := client.Post(server.URL, "test", "request body", headers, queryParams)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status 200, got %d", resp.StatusCode)
	}
}

func TestHttpClient_Post_Error_NewRequest(t *testing.T) {
	client := NewHttpClient(5000, true, nil)
	_, err := client.Post(":", "test", "body", nil, nil)
	if err == nil {
		t.Errorf("Expected error due to invalid URL")
	}
}

func TestHttpClient_Post_Error_Do(t *testing.T) {
	transportErr := errors.New("transport error")
	expectedErr := "Post \"http://example.com/test\": transport error"
	client := &httpClient{
		client: &http.Client{
			Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				return nil, transportErr
			}),
		},
	}
	_, err := http.NewRequest(http.MethodPost, "http://example.com/test", bytes.NewBufferString("body"))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}
	_, err = client.Post("http://example.com", "test", "body", nil, nil)
	if err == nil || err.Error() != expectedErr {
		t.Errorf("Expected transport error, got %v", err)
	}
}

func TestHttpClient_PostWithBytes(t *testing.T) {
	handler := func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("Expected POST method, got %s", r.Method)
		}
		if r.URL.Path != "/test" {
			t.Errorf("Expected path /test, got %s", r.URL.Path)
		}
		body, _ := ioutil.ReadAll(r.Body)
		expectedBody := []byte("byte body")
		if !bytes.Equal(body, expectedBody) {
			t.Errorf("Expected body to be %v, got %v", expectedBody, body)
		}
		if r.Header.Get("Header-Key") != "Header-Value" {
			t.Errorf("Expected header Header-Key to be Header-Value")
		}
		if r.URL.Query().Get("param") != "value" {
			t.Errorf("Expected query param param to be value")
		}
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	}
	server := httptest.NewServer(http.HandlerFunc(handler))
	defer server.Close()

	client := NewHttpClient(5000, true, nil)
	headers := map[string]string{"Header-Key": "Header-Value"}
	queryParams := map[string]string{"param": "value"}
	resp, err := client.PostWithBytes(server.URL, "test", []byte("byte body"), headers, queryParams)
	if err != nil {
		t.Fatalf("Expected no error, got %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status 200, got %d", resp.StatusCode)
	}
}

func TestHttpClient_PostWithBytes_Error_NewRequest(t *testing.T) {
	client := NewHttpClient(5000, true, nil)
	_, err := client.PostWithBytes(":", "test", []byte("body"), nil, nil)
	if err == nil {
		t.Errorf("Expected error due to invalid URL")
	}
}

func TestHttpClient_PostWithBytes_Error_Do(t *testing.T) {
	transportErr := errors.New("transport error")
	expectedErr := "Post \"http://example.com/test\": transport error"
	client := &httpClient{
		client: &http.Client{
			Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				return nil, transportErr
			}),
		},
	}
	_, err := http.NewRequest(http.MethodPost, "http://example.com/test", bytes.NewBuffer([]byte("body")))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}
	_, err = client.PostWithBytes("http://example.com", "test", []byte("body"), nil, nil)
	if err == nil || err.Error() != expectedErr {
		t.Errorf("Expected transport error, got %v", err)
	}
}

type roundTripFunc func(req *http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}
