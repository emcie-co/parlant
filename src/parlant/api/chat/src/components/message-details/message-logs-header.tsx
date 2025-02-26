import {sessionAtom} from '@/store';
import {EventInterface} from '@/utils/interfaces';
import {useAtom} from 'jotai';
import {ClassNameValue, twMerge} from 'tailwind-merge';
import HeaderWrapper from '../header-wrapper/header-wrapper';
import CopyText from '../ui/custom/copy-text';
import {X} from 'lucide-react';

const MessageLogsHeader = ({
	event,
	regenerateMessageFn,
	resendMessageFn,
	closeLogs,
	className,
}: {
	event: EventInterface | null;
	regenerateMessageFn?: (messageId: string) => void;
	resendMessageFn?: (messageId: string) => void;
	closeLogs?: VoidFunction;
	className?: ClassNameValue;
}) => {
	const [session] = useAtom(sessionAtom);
	const isCustomer = event?.source === 'customer';

	return (
		<HeaderWrapper className={twMerge('static bg-[#FBFBFB]', !event && '!border-transparent bg-[#f5f6f8]', className)}>
			{event && (
				<div className={twMerge('flex items-center justify-between w-full pe-[20px]')}>
					<div className='flex items-center gap-[12px] mb-[1px]'>
						<div
							className='group flex rounded-[5px] ms-[4px] items-center gap-[7px] py-[13px] px-[10px]'
							role='button'
							onClick={() => (event?.source === 'customer' ? resendMessageFn?.(session?.id as string) : regenerateMessageFn?.(session?.id as string))}>
							<img src={isCustomer ? 'icons/resend.svg' : 'icons/regenerate-arrow.svg'} alt='regenerate' className='block group-hover:hidden' />
							<img src={isCustomer ? 'icons/resend-hover.svg' : 'icons/regenerate-arrow-hover.svg'} alt='regenerate' className='hidden group-hover:block' />
						</div>
						<div className='group flex items-center gap-[3px] text-[14px] font-normal'>
							<CopyText preText='Message ID:' textToCopy={event.id} text={` ${event.id}`} />
						</div>
					</div>
					<div className='group'>
						<div role='button' className='p-[5px]' onClick={() => closeLogs?.()}>
							<X height={20} width={20} />
						</div>
					</div>
				</div>
			)}
		</HeaderWrapper>
	);
};

export default MessageLogsHeader;
