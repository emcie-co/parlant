import {twJoin} from 'tailwind-merge';
import {X} from 'lucide-react';
import {copy} from '@/lib/utils';
import clsx from 'clsx';
import {Log} from '@/utils/interfaces';
import {useRef} from 'react';
import Tooltip from '../ui/custom/tooltip';
import {useDialog} from '@/hooks/useDialog';
import CodeEditor from '../ui/custom/line-no-div';

const MessageLog = ({log}: {log: Log}) => {
	const {openDialog, DialogComponent, closeDialog} = useDialog();
	const ref = useRef<HTMLPreElement>(null);

	const openLogs = (text: string) => {
		const element = (
			<pre ref={ref} className='group rounded-[12px] fixed-scroll font-light font-ibm-plex-mono  border-y-[10px] border-white text-wrap text-[#333] relative overflow-auto h-[100%]'>
				<div className='invisble w-fit [justify-self:end] z-[999] group-hover:visible flex fixed top-[10px] right-[10px] justify-end'>
					<div className='flex justify-end bg-white p-[10px] gap-[20px] rounded-lg w-fit'>
						<Tooltip value='Copy' side='top'>
							<img src='icons/copy.svg' alt='' onClick={() => copy(text, ref?.current || undefined)} className='cursor-pointer' />
						</Tooltip>
						<Tooltip value='Close' side='top'>
							<X onClick={() => closeDialog()} size={18} className='cursor-pointer' />
						</Tooltip>
					</div>
				</div>
				{/* <div className='[word-break:break-all]'>{text}</div> */}
				<CodeEditor text={text}></CodeEditor>
			</pre>
		);
		openDialog('', element, {height: '90vh', width: 'min(90vw, 1200px)'});
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
			<pre className={clsx('max-w-[-webkit-fill-available] border-y-[10px] border-white group-hover:border-[#FAFAFA] font-light font-ibm-plex-mono pe-[10px] text-wrap')}>{log?.message?.trim()}</pre>
			<DialogComponent />
		</div>
	);
};

export default MessageLog;
