import Markdown from '../markdown/markdown';
import {twMerge} from 'tailwind-merge';
import Tooltip from '../ui/custom/tooltip';
import {copy} from '@/lib/utils';
import {useEffect, useState} from 'react';

const DraftBubble = ({draft = '', open = false}) => {
	const [wasOpen, setWasOpen] = useState(false);

	useEffect(() => {
		if (open) setWasOpen(true);
	}, [open]);

	return (
		<div className={twMerge('group/main flex !origin-top min-w-full overflow-hidden', !open && !wasOpen && 'h-0 opacity-0', open ? 'animate-slide-down' : wasOpen ? 'animate-slide-up' : '')}>
			<div className='text-gray-400 relative px-[22px] peer/draft py-[20px] bg-[#F5F6F8] rounded-[22px] mb-[16px] max-w-[min(560px,calc(100%-30px))] min-w-[min(560px,100%)]'>
				<Markdown className='leading-[26px]'>{draft}</Markdown>
			</div>
			<div className={twMerge('mx-[10px] self-stretch relative invisible items-center flex group-hover/main:visible peer-hover:visible hover:visible')}>
				<Tooltip value='Copy' side='top'>
					<div data-testid='copy-button' role='button' onClick={() => copy(draft || '')} className='group cursor-pointer'>
						<img src='icons/copy.svg' alt='edit' className='block opacity-50 rounded-[10px] group-hover:bg-[#EBECF0] size-[30px] p-[5px]' />
					</div>
				</Tooltip>
			</div>
		</div>
	);
};

export default DraftBubble;
