import {getDateStr} from '@/utils/date';
import {memo, ReactElement} from 'react';
import {twMerge} from 'tailwind-merge';

const DateHeader = ({date, isFirst, bgColor}: {date: string | Date; isFirst: boolean; bgColor?: string}): ReactElement => {
	return (
		<div className='flex justify-center min-h-[30px] z-[1] bg-white h-[30px] pb-[4px] mb-[14px] pt-[4px] sticky -top-[1px]'>
			<div className={twMerge('text-center flex justify-center max-w-[min(1000px,100%)] min-w-[min(1000px,100%)]', isFirst && 'pt-[1px] !mt-0', bgColor)}>
				<div className='[box-shadow:0_-0.6px_0px_0px_#F3F5F9] h-full -translate-y-[-50%] flex-1 ' />
				<div className='w-[100px] border-[0.6px] border-muted font-light text-[12px] bg-white text-[#656565] flex items-center justify-center rounded-[6px]'>{getDateStr(date)}</div>
				<div className='[box-shadow:0_-0.6px_0px_0px_#F3F5F9] h-full -translate-y-[-50%] flex-1' />
			</div>
		</div>
	);
};

export default memo(DateHeader);
