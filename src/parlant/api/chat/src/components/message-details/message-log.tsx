import {twJoin} from 'tailwind-merge';
import {X} from 'lucide-react';
import {copy} from '@/lib/utils';
import clsx from 'clsx';
import {Log} from '@/utils/interfaces';
import {useRef} from 'react';
import Tooltip from '../ui/custom/tooltip';
import {useDialog} from '@/hooks/useDialog';

const MessageLog = ({log}: {log: Log}) => {
	const {openDialog, DialogComponent, closeDialog} = useDialog();
	const ref = useRef<HTMLPreElement>(null);

	const openLogs = (text: string) => {
		const element = (
			<pre ref={ref} className='group font-light font-ibm-plex-mono px-[30px] py-[10px] text-wrap text-[#333] relative overflow-auto h-[100%]'>
				<div className='invisble group-hover:visible flex sticky top-[10px] right-[20px] justify-end'>
					<div className='flex justify-end bg-white p-[10px] gap-[20px] rounded-lg'>
						<Tooltip value='Copy' side='top'>
							<img src='icons/copy.svg' alt='' onClick={() => copy(text, ref?.current || undefined)} className='cursor-pointer' />
						</Tooltip>
						<Tooltip value='Close' side='top'>
							<X onClick={() => closeDialog()} size={18} className='cursor-pointer' />
						</Tooltip>
					</div>
				</div>
				<div>{text}</div>
			</pre>
		);
		openDialog('', element, {height: '90vh', width: '90vw'});
	};

	return (
		<div className={twJoin('flex max-h-[200px] w-full overflow-hidden group relative font-ubuntu-mono gap-[5px] px-[20px] text-[14px] transition-all  hover:bg-[#FAFAFA]')}>
			<div className='absolute hidden z-10 group-hover:flex right-[10px] top-[10px] gap-[5px]'>
				<Tooltip value='Copy' side='top'>
					<div onClick={() => copy(log?.message || '')} className='cursor-pointer size-[28px] flex justify-center items-center bg-white hover:bg-[#F3F5F9] border border-[#EEEEEE] hover:border-[#E9EBEF] rounded-[6px]'>
						<img src='icons/copy.svg' alt='' />
					</div>
				</Tooltip>
				<Tooltip value='Expand' side='top'>
					<div onClick={() => openLogs(log?.message || '')} className='cursor-pointer size-[28px] flex justify-center items-center bg-white hover:bg-[#F3F5F9] border border-[#EEEEEE] hover:border-[#E9EBEF] rounded-[6px]'>
						<img src='icons/expand.svg' alt='' />
					</div>
				</Tooltip>
			</div>
			<pre className={clsx('max-w-[-webkit-fill-available] font-light font-ibm-plex-mono pe-[10px] text-wrap')}>
				{log?.level ? `[${log.level}]` : ''}
				{log?.message}
			</pre>
			<DialogComponent />
		</div>
	);
};

export default MessageLog;
