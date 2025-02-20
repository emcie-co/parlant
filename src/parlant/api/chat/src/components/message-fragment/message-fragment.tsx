import Tooltip from '../ui/custom/tooltip';
import {copy} from '@/lib/utils';
import {twMerge} from 'tailwind-merge';

const TooltipComponent = ({fragmentId}: {fragmentId: string}) => {
	return (
		<div className='group flex gap-[4px] text-[#CDCDCD] hover:text-[#151515]' role='button' onClick={() => copy(fragmentId)}>
			<div>Fragment ID: {fragmentId}</div>
			<img src='icons/copy.svg' alt='' className='invisible group-hover:visible' />
		</div>
	);
};

const MessageFragment = ({fragment}: {fragment: {id: string; value: string}}) => {
	return (
		<Tooltip value={<TooltipComponent fragmentId={fragment.id} />} side='top' align='start' className='rounded-[6px] rounded-bl-[0px] ml-[23px] -mb-[10px] font-medium font-ubuntu-sans'>
			<div className='group [word-break:break-word] rounded-[8px] hover:bg-[#F5F6F8] hover:border-[#EDEDED] border border-transparent flex gap-[17px] items-start text-[#656565] py-[8px] ps-[15px] pe-[38px]'>
				<img src='icons/puzzle.svg' alt='' className='group-hover:hidden mt-[4px] w-[16px] min-w-[16px]' />
				<img src='icons/puzzle-hover.svg' alt='' className='hidden group-hover:block mt-[4px] w-[16px] min-w-[16px]' />
				<div className={twMerge('invisible', fragment?.value && 'visible')}>{fragment?.value || 'loading'}</div>
			</div>
		</Tooltip>
	);
};

export default MessageFragment;
