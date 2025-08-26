import {ReactNode} from 'react';
import {toast} from 'sonner';
import {twMerge} from 'tailwind-merge';
import {spaceClick} from '@/utils/methods';
import {fallbackCopyText} from '@/lib/utils';

interface Props {
	text: string;
	textToCopy?: string;
	preText?: string;
	className?: string;
	element?: HTMLElement;
}

export default function CopyText({text, textToCopy, preText, className, element}: Props): ReactNode {
	if (!textToCopy) textToCopy = text;

	const copyClicked = (e: React.MouseEvent) => {
		e.stopPropagation();
		if (navigator.clipboard && navigator.clipboard.writeText) {
			navigator.clipboard
				.writeText(textToCopy)
				.then(() => toast.info(`Copied text: ${textToCopy}`))
				.catch(() => {
					fallbackCopyText(textToCopy, element);
				});
		} else {
			fallbackCopyText(textToCopy, element);
		}
	};

	return (
		<div className={twMerge('group flex gap-[6px] items-center cursor-pointer text-[#A9A9A9] text-[15px] font-light', className)} onKeyDown={spaceClick} onClick={copyClicked}>
			<div className='flex items-center gap-[6px]'>
				{preText && <span className='font-semibold'>{preText}</span>}
				<span className='group-hover:text-[#656565]'>{text}</span>
			</div>
			<div className='copy-icon hidden group-hover:block group-hover:text-[#656565]' role='button' tabIndex={0}>
				<img src='icons/copy.svg' alt='' />
			</div>
		</div>
	);
}
