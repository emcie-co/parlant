import {EventInterface, ServerStatus} from './interfaces';

export const rootTraceId = (traceId: string): string => traceId.split('::')[0];

export const withLatestTraceStatus = (
	message: EventInterface,
	traceEvents: EventInterface[],
	fallbackStatus?: ServerStatus
): EventInterface => {
	const statusEvent = traceEvents.findLast((event) => event.kind === 'status');
	const serverStatus = statusEvent?.data?.status || fallbackStatus || message.serverStatus;
	const result: EventInterface = {
		...message,
		serverStatus,
		metadata: message.metadata ? {...message.metadata} : undefined,
	};

	if (serverStatus === 'error') {
		result.error = statusEvent?.data?.data?.exception;
		const diagnosticReportUri = statusEvent?.metadata?.__parlant__?.diagnostic_report_uri;
		if (diagnosticReportUri) {
			result.metadata = {
				...result.metadata,
				__parlant__: {
					...result.metadata?.__parlant__,
					schema_version: statusEvent?.metadata?.__parlant__?.schema_version || 1,
					diagnostic_report_uri: diagnosticReportUri,
				},
			};
		}
	}

	return result;
};
