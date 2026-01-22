/**
 * LegalGPT SSE Client
 * Handles real-time updates from the server via Server-Sent Events
 */

class SSEClient {
    constructor(url, options = {}) {
        this.url = url;
        this.eventSource = null;
        this.reconnectDelay = options.reconnectDelay || 3000;
        this.maxRetries = options.maxRetries || 10;
        this.retries = 0;
        this.handlers = {};
        this.onConnect = options.onConnect || (() => {});
        this.onDisconnect = options.onDisconnect || (() => {});
        this.onError = options.onError || console.error;
    }

    connect() {
        if (this.eventSource) {
            this.eventSource.close();
        }

        this.eventSource = new EventSource(this.url);

        this.eventSource.onopen = () => {
            console.log('SSE connected to', this.url);
            this.retries = 0;
            this.onConnect();
        };

        this.eventSource.onerror = (error) => {
            console.error('SSE error:', error);
            this.onDisconnect();
            this.eventSource.close();

            if (this.retries < this.maxRetries) {
                this.retries++;
                console.log(`Reconnecting in ${this.reconnectDelay}ms (attempt ${this.retries}/${this.maxRetries})`);
                setTimeout(() => this.connect(), this.reconnectDelay);
            } else {
                this.onError('Max retries reached');
            }
        };

        // Register event handlers
        for (const [eventType, handler] of Object.entries(this.handlers)) {
            this.eventSource.addEventListener(eventType, (event) => {
                try {
                    const data = JSON.parse(event.data);
                    handler(data);
                } catch (e) {
                    handler(event.data);
                }
            });
        }
    }

    on(eventType, handler) {
        this.handlers[eventType] = handler;

        // If already connected, add listener
        if (this.eventSource) {
            this.eventSource.addEventListener(eventType, (event) => {
                try {
                    const data = JSON.parse(event.data);
                    handler(data);
                } catch (e) {
                    handler(event.data);
                }
            });
        }

        return this;
    }

    disconnect() {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SSEClient;
}
