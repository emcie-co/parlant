import {useVirtualizer} from '@tanstack/react-virtual';
import {Log} from '@/utils/interfaces';
import MessageLog from './message-log';

interface Props {
	messagesRef: React.RefObject<HTMLDivElement>;
	filteredLogs: Log[];
}

const MessageLogs = ({messagesRef, filteredLogs}: Props) => {
	// Virtualize: only the visible rows (plus a small buffer) are mounted, so opening
	// a message's logs is fast regardless of how many there are. Each row is heavy (it
	// instantiates its own dialog/tooltips), so rendering thousands at once is what
	// froze the UI.
	const virtualizer = useVirtualizer({
		count: filteredLogs.length,
		getScrollElement: () => messagesRef.current,
		estimateSize: () => 48, // the row's min-height; each row is remeasured below
		overscan: 12,
	});

	return (
		<div className='p-[6px] overflow-hidden h-[calc(100%-12px)] rounded-[6px]'>
			<div className='pt-0 flex-1 border bg-white h-full rounded-[3px]'>
				<div className='flex items-center min-h-[48px] text-[14px] font-medium border-b border-[#EDEFF3]'>
					<div className='w-[86px] border-e border-[#EDEFF3] min-h-[48px] flex items-center ps-[10px]'>Level</div>
					<div className='flex-1 ps-[10px]'>Message</div>
				</div>
				<div ref={messagesRef} className='rounded-[8px] h-[calc(100%-60px)] overflow-auto bg-white fixed-scroll text-[14px] font-normal'>
					<div style={{height: virtualizer.getTotalSize(), width: '100%', position: 'relative'}}>
						{virtualizer.getVirtualItems().map((virtualRow) => {
							const log = filteredLogs[virtualRow.index];
							return (
								<div
									key={virtualRow.key}
									data-index={virtualRow.index}
									ref={virtualizer.measureElement}
									style={{position: 'absolute', top: 0, left: 0, width: '100%', transform: `translateY(${virtualRow.start}px)`}}
									className='flex group hover:bg-[#FAFAFA] min-h-[48px] border-t border-[#EDEFF3] font-ibm-plex-mono items-stretch'>
									<div className='min-w-[86px] w-[86px] border-e border-[#EDEFF3] min-h-[48px] flex ps-[10px] pt-[10px] capitalize'>{log.level?.toLowerCase()}</div>
									<MessageLog log={log} />
								</div>
							);
						})}
					</div>
				</div>
			</div>
		</div>
	);
};
export default MessageLogs;
