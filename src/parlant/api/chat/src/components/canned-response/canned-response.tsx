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

const CannedResponse = ({cannedResponse: cannedResponse}: {cannedResponse: {id: string; value: string}}) => {
	const [id, value] = cannedResponse?.value || ['', ''];
	return (
		<div className='group relative flex justify-between group min-h-[40px] bg-white hover:bg-[#FAFAFA]'>
			<div className='group [word-break:break-word] w-full flex gap-[17px] font-light [&:first-child]:rounded-t-[3px] items-start text-[#656565] py-[8px] ps-[15px] pe-[38px]'>
				<img src='icons/puzzle.svg' alt='' className='mt-[4px] w-[16px] min-w-[16px]' />
				<div className={twMerge('invisible', value && 'visible')}>{value || 'loading'}</div>
			</div>
			<Tooltip value='Copy' side='top'>
				<div
					onClick={(e) => copy(id || '', e.currentTarget)}
					className='hidden absolute right-[10px] top-[8px] cursor-pointer size-[28px] group-hover:flex justify-center items-center bg-white hover:bg-[#F3F5F9] border border-[#EEEEEE] hover:border-[#E9EBEF] rounded-[6px]'>
					<img src='icons/copy.svg' alt='' />
				</div>
			</Tooltip>
		</div>
		// <Tooltip value={<TooltipComponent fragmentId={fragment.id} />} side='top' align='start' className='ml-[23px] -mb-[10px] font-medium font-inter'>
		// </Tooltip>
	);
};

export default CannedResponse;
