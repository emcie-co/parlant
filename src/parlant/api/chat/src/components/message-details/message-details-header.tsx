import {dialogAtom, sessionAtom} from '@/store';
import {EventInterface} from '@/utils/interfaces';
import {useAtom} from 'jotai';
import {ClassNameValue, twMerge} from 'tailwind-merge';
import HeaderWrapper from '../header-wrapper/header-wrapper';
import {Flag, X} from 'lucide-react';
import {Button} from '../ui/button';
import FlagMessage from './flag-message';
import {useEffect, useState} from 'react';
import {getItemFromIndexedDB} from '@/lib/utils';

const MessageDetailsHeader = ({
	event,
	sameCorrelationMessages,
	regenerateMessageFn,
	resendMessageFn,
	closeLogs,
	className,
	flaggedChanged,
}: {
	event: EventInterface | null;
	sameCorrelationMessages?: EventInterface[];
	regenerateMessageFn?: (messageId: string) => void;
	resendMessageFn?: (messageId: string) => void;
	closeLogs?: VoidFunction;
	className?: ClassNameValue;
	flaggedChanged?: (flagged: boolean) => void;
}) => {
	const [session] = useAtom(sessionAtom);
	const [dialog] = useAtom(dialogAtom);
	const isCustomer = event?.source === 'customer';
	const [messageFlag, setMessageFlag] = useState<any>(null);
	const [refreshFlag, setRefreshFlag] = useState(false);

	useEffect(() => {
		const flag = getItemFromIndexedDB('Parlant-flags', 'message_flags', event?.correlation_id as string, {name: 'sessionIndex', keyPath: 'sessionId'});
		if (flag) {
			flag.then((f) => {
				setMessageFlag((f as {flagValue: string})?.flagValue);
				flaggedChanged?.(!!(f as {flagValue: string})?.flagValue);
			});
		}
	}, [event, refreshFlag]);

	return (
		<HeaderWrapper className={twMerge('static', !event && '!border-transparent bg-[#f5f6f8]', className)}>
			{event && (
				<div className={twMerge('flex items-center justify-between w-full pe-[12px]')}>
					<div className='flex'>
						<div role='button' className='p-[5px] pe-[10px]' onClick={() => closeLogs?.()}>
							<X height={25} width={25} />
						</div>
					</div>
					<div className='flex items-center gap-[12px] mb-[1px]'>
						{!isCustomer && (
							<Button
								className={twMerge('gap-1', messageFlag && 'border-[#9B0360] !text-[#9B0360]')}
								variant='outline'
								onClick={() =>
									dialog.openDialog('Flag Response', <FlagMessage existingFlagValue={messageFlag || ''} events={sameCorrelationMessages || [event]} sessionId={session?.id as string} onFlag={() => setRefreshFlag(!refreshFlag)} />, {
										width: '600px',
										height: '636px',
									})
								}>
								<Flag color={messageFlag ? '#9B0360' : 'black'} size={16} />
								<div>{messageFlag ? 'View Comment' : 'Flag'}</div>
							</Button>
						)}
						<div
							className='group bg-[#006E53] [box-shadow:0px_2px_4px_0px_#00403029,0px_1px_5.5px_0px_#006E5329] hover:bg-[#005C3F] flex  h-[38px] rounded-[5px] ms-[4px] items-center gap-[7px] py-[13px] px-[10px]'
							role='button'
							onClick={() => (event?.source === 'customer' ? resendMessageFn?.(session?.id as string) : regenerateMessageFn?.(session?.id as string))}>
							<img src='icons/regenerate.svg' alt='regenerate' className='block' />
							<div className='text-white text-[14px] font-normal'>{isCustomer ? 'Resend' : 'Regenerate'}</div>
							{/* <img src={isCustomer ? 'icons/resend-hover.svg' : 'icons/regenerate-arrow-hover.svg'} alt='regenerate' className='hidden group-hover:block' /> */}
						</div>
					</div>
				</div>
			)}
		</HeaderWrapper>
	);
};

export default MessageDetailsHeader;
