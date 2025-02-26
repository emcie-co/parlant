import {Log} from '@/utils/interfaces';
import MessageLog from './message-logs/message-log';

interface Props {
	messagesRef: React.RefObject<HTMLDivElement>;
	filteredLogs: Log[];
}

const MessageLogs = ({messagesRef, filteredLogs}: Props) => {
	return (
		<div className='bg-[#EBECF0] p-[14px] pt-0 h-auto overflow-auto flex-1'>
			<div className='flex items-center'>
				<div className='w-[86px]'>Level</div>
				<div>Message</div>
			</div>
			<div ref={messagesRef} className='rounded-[8px] border-[10px] border-white h-full overflow-auto bg-white fixed-scroll'>
				{filteredLogs.map((log, i) => (
					<div className='flex items-center min-h-[48px] border-t'>
						<div className='min-w-[86px] w-[86px]'>{log.level}</div>
						<MessageLog key={i} log={log} />
					</div>
				))}
			</div>
		</div>
	);
};
export default MessageLogs;
