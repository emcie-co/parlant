import {Log} from '@/utils/interfaces';
import MessageLog from './message-log';

interface Props {
	messagesRef: React.RefObject<HTMLDivElement>;
	filteredLogs: Log[];
}

const MessageLogs = ({messagesRef, filteredLogs}: Props) => {
	return (
		<div className='p-[6px] overflow-hidden h-[calc(100%-12px)] rounded-[6px]'>
			<div className='pt-0 flex-1 border bg-white h-full rounded-[3px]'>
				<div className='flex items-center min-h-[48px] text-[14px] font-medium border-b border-[#EDEFF3]'>
					<div className='w-[86px] border-e border-[#EDEFF3] min-h-[48px] flex items-center ps-[10px]'>Level</div>
					<div className='flex-1 ps-[10px]'>Message</div>
				</div>
				<div ref={messagesRef} className='rounded-[8px] h-[calc(100%-60px)] overflow-auto bg-white fixed-scroll text-[14px] font-normal'>
					{filteredLogs.map((log, i) => (
						<div key={i} className='flex group hover:bg-[#FAFAFA] min-h-[48px] border-t border-[#EDEFF3] font-ibm-plex-mono [&:last-child]:border-b [&:first-child]:border-[0px] items-stretch'>
							<div className='min-w-[86px] w-[86px] border-e border-[#EDEFF3] min-h-[48px] flex ps-[10px] pt-[10px] capitalize'>{log.level?.toLowerCase()}</div>
							<MessageLog log={log} />
						</div>
					))}
				</div>
			</div>
		</div>
	);
};
export default MessageLogs;
