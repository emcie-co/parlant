export interface AgentInterface {
    id: string;
    name: string;
};

export interface CustomerInterface {
    id: string;
    name: string;
};

export type ServerStatus = 'pending' | 'error' | 'accepted' | 'acknowledged' | 'processing' | 'typing' | 'ready';
type eventSource = 'customer' | 'customer_ui' | 'human_agent' | 'human_agent_on_behalf_of_ai_agent' | 'ai_agent' | 'system';

export interface EventInterface {
    id?: string;
    source: eventSource;
    kind: 'status' | 'message';
    correlation_id: string;
    serverStatus: ServerStatus;
    offset: number;
    creation_utc: Date;
    data: {
        status?: ServerStatus;
        message: string;
    };
};

export interface SessionInterface {
    id: string;
    title: string;
    customer_id: string;
    agent_id: string;
    creation_utc: string;
};