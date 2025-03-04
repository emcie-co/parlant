import Tooltip from '../ui/custom/tooltip';
import {copy} from '@/lib/utils';
import {twMerge} from 'tailwind-merge';

// const TooltipComponent = ({fragmentId}: {fragmentId: string}) => {
// 	return (
// 		<div className='group flex gap-[4px] text-[#CDCDCD] hover:text-[#151515]' role='button' onClick={() => copy(fragmentId)}>
// 			<div>Fragment ID: {fragmentId}</div>
// 			<img src='icons/copy.svg' alt='' className='invisible group-hover:visible' />
// 		</div>
// 	);
// };

const MessageFragment = ({fragment}: {fragment: {id: string; value: string}}) => {
	return (
		<div className='group flex justify-between group min-h-[40px]'>
			<div className='group [word-break:break-word] w-full bg-white group-hover:bg-[#FAFAFA] flex gap-[17px] [&:first-child]:rounded-t-[3px] items-start text-[#656565] py-[8px] ps-[15px] pe-[38px]'>
				<img src='icons/puzzle.svg' alt='' className='mt-[4px] w-[16px] min-w-[16px]' />
				<div className={twMerge('invisible', fragment?.value && 'visible')}>{fragment?.value || 'loading'}</div>
			</div>
			<Tooltip value='Copy' side='top'>
				<div
					onClick={(e) => copy(fragment.id || '', e.currentTarget)}
					className='hidden me-[10px] mt-[6px] cursor-pointer size-[28px] group-hover:flex justify-center items-center bg-white hover:bg-[#F3F5F9] border border-[#EEEEEE] hover:border-[#E9EBEF] rounded-[6px]'>
					<img src='icons/copy.svg' alt='' />
				</div>
			</Tooltip>
		</div>
		// <Tooltip value={<TooltipComponent fragmentId={fragment.id} />} side='top' align='start' className='ml-[23px] -mb-[10px] font-medium font-ubuntu-sans'>
		// </Tooltip>
	);
};

export default MessageFragment;
