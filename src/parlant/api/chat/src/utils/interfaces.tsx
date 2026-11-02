export interface AgentInterface {
	id: string;
	name: string;
}

export interface CustomerInterface {
	id: string;
	name: string;
}

export interface Log {
	level: 'CRITICAL' | 'ERROR' | 'WARNING' | 'INFO' | 'DEBUG' | 'TRACE';
	trace_id: string;
	message: string;
	timestamp: number;
}

export type ServerStatus = 'pending' | 'error' | 'accepted' | 'acknowledged' | 'processing' | 'typing' | 'ready' | 'cancelled';
type eventSource = 'customer' | 'customer_ui' | 'human_agent' | 'human_agent_on_behalf_of_ai_agent' | 'ai_agent' | 'system';

export interface EventInterface {
	id?: string;
	source: eventSource;
	kind: 'status' | 'message';
	trace_id: string;
	serverStatus: ServerStatus;
	sessionId?: string;
	error?: string;
	offset: number;
	creation_utc: Date;
	data: {
		participant?: { display_name?: string }
		status?: ServerStatus;
		draft?: string;
		canned_responses?: string[];
		message: string;
		data?: { exception?: string, stage?: string };
		groups?: string;
		chunks?: (string | null)[];
	};
	/**
	 * Out-of-band annotations on the event. Internal Parlant annotations are
	 * namespaced below `__parlant__`; user-authored keys remain siblings.
	 * `__parlant__.voice.spoken_text` records the part
	 * of an agent message that was actually played before an interruption —
	 * absent means fully delivered, empty means never heard. Read it through
	 * `deliveredMessageText`, never directly.
	 */
	metadata?: {
		__parlant__?: {
			schema_version: number;
			voice?: { spoken_text?: string; speculative?: boolean };
			diagnostic_report_uri?: string;
			source?: string;
			reasoning?: boolean;
			provider_data?: Record<string, unknown>;
		};
		[key: string]: unknown;
	};
	index?: number;
}

export interface SessionInterface {
	id: string;
	title: string;
	customer_id: string;
	agent_id: string;
	creation_utc: string;
	modified_utc?: string;
}

export interface SessionCsvInterface {
	Source: 'AI Agent' | 'Customer';
	Participant: string;
	Timestamp: Date;
	Message: string;
	Draft: string;
	Groups: string;
	Flag: string;
	'Trace ID': string;
}
