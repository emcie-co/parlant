import {useState} from 'react';
import {ClassNameValue, twMerge} from 'tailwind-merge';
import MessageFragment from '../message-fragment/message-fragment';
import ErrorBoundary from '../error-boundary/error-boundary';

export interface Fragment {
	id: string;
	value: string;
}

const MessageFragments = ({fragments, className}: {fragments: {id: string; value: string}[]; className?: ClassNameValue}) => {
	const [isOpen, setIsOpen] = useState(false);

	const onToggle = (e: any) => {
		setIsOpen(e.target.open);
	};

	return (
		<details onToggle={onToggle} open className={twMerge(isOpen && 'bg-[#F5F6F8]', className)}>
			<summary className={twMerge('h-[34px] flex items-center justify-between ms-[24px] me-[30px] cursor-pointer text-[16px] bg-[#FBFBFB] hover:bg-white text-[#656565] hover:text-[#151515]', isOpen && '!bg-[#F5F6F8] !text-[#656565]')}>
				<span>Fragments</span>
				<img src='icons/arrow-down.svg' alt='' style={{rotate: isOpen ? '0deg' : '180deg'}} />
			</summary>
			<div className='p-[14px] pt-[10px]'>
				<div className='rounded-[14px] bg-white p-[10px]'>
					<div className='overflow-auto fixed-scroll max-h-[308px]'>
						<ErrorBoundary component={<div>Could not load fragments</div>}>
							{fragments.map((fragment) => (
								<MessageFragment key={fragment.id} fragment={fragment} />
							))}
						</ErrorBoundary>
					</div>
				</div>
			</div>
		</details>
	);
};

export default MessageFragments;
