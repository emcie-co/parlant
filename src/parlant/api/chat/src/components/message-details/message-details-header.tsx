import {sessionAtom} from '@/store';
import {EventInterface} from '@/utils/interfaces';
import {useAtom} from 'jotai';
import {ClassNameValue, twMerge} from 'tailwind-merge';
import HeaderWrapper from '../header-wrapper/header-wrapper';
import CopyText from '../ui/custom/copy-text';
import {X} from 'lucide-react';

const MessageDetailsHeader = ({
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
		<HeaderWrapper className={twMerge('static', !event && '!border-transparent bg-[#f5f6f8]', className)}>
			{event && (
				<div className={twMerge('flex items-center justify-between w-full pe-[12px]')}>
					<div className='flex ps-[16px]'>
						<div role='button' className='p-[5px] pe-[10px]' onClick={() => closeLogs?.()}>
							<X height={25} width={25} />
						</div>
						<div className='group flex items-center gap-[3px] text-[14px] font-normal'>
							<CopyText preText='Message ID:' textToCopy={event.id} text={` ${event.id}`} />
						</div>
					</div>
					<div className='flex items-center gap-[12px] mb-[1px]'>
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
