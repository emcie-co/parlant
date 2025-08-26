/* eslint-disable react-hooks/exhaustive-deps */
import {ReactElement, useEffect, useRef, useState} from 'react';
import {EventInterface} from '@/utils/interfaces';
import Spacer from '../ui/custom/spacer';
import {twMerge} from 'tailwind-merge';
import {Textarea} from '../ui/textarea';
import {Button} from '../ui/button';
import {useAtom} from 'jotai';
import {sessionAtom} from '@/store';
import MessageBubble from './message-bubble';

interface Props {
	event: EventInterface;
	sameCorrelationMessages?: EventInterface[];
	isContinual: boolean;
	isRegenerateHidden?: boolean;
	isFirstMessageInDate?: boolean;
	flagged?: string;
	flaggedChanged?: (flagged: string) => void;
	showLogsForMessage?: EventInterface | null;
	regenerateMessageFn?: (sessionId: string) => void;
	resendMessageFn?: (sessionId: string, text?: string) => void;
	showLogs: (event: EventInterface) => void;
	setIsEditing?: React.Dispatch<React.SetStateAction<boolean>>;
}



const MessageEditing = ({event, resendMessageFn, setIsEditing}: Props) => {
	const ref = useRef<HTMLDivElement>(null);
	const textArea = useRef<HTMLTextAreaElement>(null);
	const [textValue, setTextValue] = useState(event?.data?.message || '');
	const [session] = useAtom(sessionAtom);

	useEffect(() => {
		textArea?.current?.select();
	}, [textArea?.current]);

	useEffect(() => {
		ref?.current?.scrollIntoView({behavior: 'smooth', block: 'nearest'});
	}, [ref?.current]);

	return (
		<div ref={ref} className='w-full p-[16px] ps-[6px] pe-[6px] rounded-[16px] max-w-[min(560px,90%)] rounded-br-none border origin-bottom bg-[#f5f6f8] ' style={{transformOrigin: 'bottom'}}>
			<Textarea ref={textArea} className='[direction:ltr] resize-none h-[120px] pe-[108px] !ring-0 !ring-offset-0 border-none ps-[22px] bg-[#f5f6f8]' onChange={(e) => setTextValue(e.target.value)} defaultValue={textValue} />
			<div className='pt-[10px] flex justify-end gap-[10px] pe-[12px] [direction:ltr]'>
				<Button variant='ghost' onClick={() => setIsEditing?.(false)} className='rounded-[10px] hover:bg-white'>
					Cancel
				</Button>
				<Button
					disabled={!textValue?.trim() || textValue?.trim() === event?.data?.message}
					className='rounded-[10px]'
					onClick={() => {
						resendMessageFn?.(session?.id || '', textValue?.trim());
						setIsEditing?.(false);
					}}>
					Apply
				</Button>
			</div>
		</div>
	);
};

function Message({event, isFirstMessageInDate, isContinual, showLogs, showLogsForMessage, resendMessageFn, flagged, flaggedChanged, sameCorrelationMessages}: Props): ReactElement {
	const [isEditing, setIsEditing] = useState(false);
	return (
		<div className={twMerge(isEditing && '[direction:rtl] flex justify-center')}>
			<div
				className={twMerge(
					'flex py-[3px] mx-0 mb-1 w-full justify-between animate-fade-in scrollbar',
					isEditing && 'flex-1 flex justify-start max-w-[1000px] items-end w-[calc(100%-412px)] max-[2100px]:w-[calc(100%-200px)] self-end max-[1700px]:w-[calc(100%-40px)]'
				)}>
				<Spacer />
				{isEditing ? (
					<MessageEditing resendMessageFn={resendMessageFn} setIsEditing={setIsEditing} event={event} isContinual={isContinual} showLogs={showLogs} showLogsForMessage={showLogsForMessage} />
				) : (
					<MessageBubble isFirstMessageInDate={isFirstMessageInDate} setIsEditing={setIsEditing} event={event} isContinual={isContinual} showLogs={showLogs} showLogsForMessage={showLogsForMessage} flagged={flagged} flaggedChanged={flaggedChanged} sameCorrelationMessages={sameCorrelationMessages} />
				)}
				<Spacer />
			</div>
		</div>
	);
}

export default Message;
