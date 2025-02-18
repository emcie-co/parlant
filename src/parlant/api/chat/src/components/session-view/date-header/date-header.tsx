import {getDateStr} from '@/utils/date';
import {ReactElement} from 'react';
import {twMerge} from 'tailwind-merge';

const DateHeader = ({date, isFirst, bgColor}: {date: string | Date; isFirst: boolean; bgColor?: string}): ReactElement => {
	return (
		<div className={twMerge('text-center flex min-h-[30px] z-[1] bg-main h-[30px] pb-[4px] ps-[10px] mb-[60px] pt-[4px] mt-[76px] sticky -top-[1px]', isFirst && 'pt-[1px] !mt-0', bgColor)}>
			<div className='[box-shadow:0_-0.6px_0px_0px_#EBECF0] h-full -translate-y-[-50%] flex-1 ' />
			<div className='w-[136px] border-[0.6px] border-muted font-light text-[12px] bg-white text-[#656565] flex items-center justify-center rounded-[6px]'>{getDateStr(date)}</div>
			<div className='[box-shadow:0_-0.6px_0px_0px_#EBECF0] h-full -translate-y-[-50%] flex-1' />
		</div>
	);
};

export default DateHeader;
