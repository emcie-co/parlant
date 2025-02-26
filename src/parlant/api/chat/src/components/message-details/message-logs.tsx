import {Log} from '@/utils/interfaces';
import MessageLog from './message-log';

interface Props {
	messagesRef: React.RefObject<HTMLDivElement>;
	filteredLogs: Log[];
}

const MessageLogs = ({messagesRef, filteredLogs}: Props) => {
	return (
		<div className='p-[6px] bg-green-light overflow-hidden'>
			<div className='pt-0 flex-1 border bg-white h-full'>
				<div className='flex items-center min-h-[48px] text-[14px] font-medium border-b'>
					<div className='w-[86px] border-e min-h-[48px] flex items-center ps-[10px] pt-[8px]'>Level</div>
					<div className='flex-1 ps-[10px] pt-[8px]'>Message</div>
				</div>
				<div ref={messagesRef} className='rounded-[8px] h-[calc(100%-50px)] overflow-auto bg-white fixed-scroll text-[14px] font-normal'>
					{filteredLogs.map((log, i) => (
						<div key={i} className='flex min-h-[48px] border-t [&:first-child]:border-[0px] items-stretch'>
							<div className='min-w-[86px] w-[86px] border-e min-h-[48px] flex ps-[10px] pt-[10px] capitalize'>{log.level?.toLowerCase()}</div>
							<MessageLog log={log} />
						</div>
					))}
				</div>
			</div>
		</div>
	);
};
export default MessageLogs;
