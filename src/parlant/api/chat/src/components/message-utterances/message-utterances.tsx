import {useState} from 'react';
import {ClassNameValue, twMerge} from 'tailwind-merge';
import MessageUtterance from '../message-utterance/message-utterance';
import ErrorBoundary from '../error-boundary/error-boundary';

export interface Utterance {
	id: string;
	value: string;
}

const MessageUtterances = ({utterances: utterances, className}: {utterances: {id: string; value: string}[]; className?: ClassNameValue}) => {
	const [isOpen, setIsOpen] = useState(false);

	const onToggle = (e: any) => {
		setIsOpen(e.target.open);
	};

	return (
		<details onToggle={onToggle} open className={twMerge('max-h-[50%]', className)}>
			<summary className={twMerge('h-[34px] bg-white flex items-center text-[#282828] justify-between ms-[24px] me-[30px] cursor-pointer text-[16px] ')}>
				<span>Utterances</span>
				<img src='icons/arrow-down.svg' alt='' style={{rotate: isOpen ? '0deg' : '180deg'}} />
			</summary>
			<div className='p-[14px] pb-0 pt-[10px]'>
				<div className='rounded-[14px]'>
					<div className='overflow-auto fixed-scroll max-h-[308px] border-[6px] border-[#F5F9F7] bg-[#F5F9F7] rounded-[10px]'>
						<ErrorBoundary component={<div>Could not load utterances</div>}>
							{utterances.map((utterance) => (
								<MessageUtterance key={utterance.id} utterance={utterance} />
							))}
						</ErrorBoundary>
					</div>
				</div>
			</div>
		</details>
	);
};

export default MessageUtterances;
