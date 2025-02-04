import {EventInterface, Log} from '@/utils/interfaces';
import {Copy, Fullscreen, Plus, X} from 'lucide-react';
import React, {ReactNode, useEffect, useRef, useState} from 'react';
import {getMessageLogs, getMessageLogsWithFilters} from '@/utils/logs';
import {ClassNameValue, twJoin, twMerge} from 'tailwind-merge';
import clsx from 'clsx';
import HeaderWrapper from '../header-wrapper/header-wrapper';
import {useLocalStorage} from '@/hooks/useLocalStorage';
import LogFilters from '../log-filters/log-filters';
import {useAtom} from 'jotai';
import {dialogAtom, sessionAtom} from '@/store';
import CopyText from '../ui/custom/copy-text';
import Tooltip from '../ui/custom/tooltip';
import {copy} from '@/lib/utils';

interface Filter {
	id: number;
	name: string;
	def: {level?: string; types?: string[]} | null;
}

interface FilterTabsFilterProps {
	filterTabs: Filter[];
	setCurrFilterTabs: React.Dispatch<React.SetStateAction<number | null>>;
	setFilterTabs: any;
	currFilterTabs: number | null;
}

const Header = ({event, regenerateMessageFn, closeLogs, className}: {event: EventInterface | null; regenerateMessageFn?: (messageId: string) => void; closeLogs?: VoidFunction; className?: ClassNameValue}) => {
	const [session] = useAtom(sessionAtom);
	return (
		<HeaderWrapper className={twMerge('static bg-[#FBFBFB]', !event && '!border-transparent bg-[#f5f6f8]', className)}>
			{event && (
				<div className={twMerge('flex items-center justify-between w-full pe-[20px]')}>
					<div className='flex items-center gap-[12px] mb-[1px]'>
						<div className='group flex rounded-[5px] ms-[4px] items-center gap-[7px] py-[13px] px-[10px]' role='button' onClick={() => regenerateMessageFn?.(session?.id as string)}>
							<img src='icons/regenerate-arrow.svg' alt='regenerate' className='block group-hover:hidden' />
							<img src='icons/regenerate-arrow-hover.svg' alt='regenerate' className='hidden group-hover:block' />
							{/* <p className='font-medium text-[15px]'>Regenerate Message</p> */}
						</div>
						<div className='group flex items-center gap-[3px] text-[14px] font-normal'>
							<CopyText preText='Message ID:' textToCopy={event.id} text={` ${event.id}`} />
						</div>
					</div>
					<div className='group'>
						<div role='button' className='p-[5px]' onClick={() => closeLogs?.()}>
							<X height={20} width={20} />
						</div>
					</div>
				</div>
			)}
		</HeaderWrapper>
	);
};

const FilterTabs = ({filterTabs, setCurrFilterTabs, setFilterTabs, currFilterTabs}: FilterTabsFilterProps) => {
	const [isEditing, setIsEditing] = useState(false);
	const [inputVal, setInputVal] = useState('');

	const deleteFilterTab = (id: number, index: number) => {
		const filteredTabs = filterTabs.filter((t) => t.id !== id);
		setFilterTabs(filteredTabs);

		if (currFilterTabs === id) {
			const newTab = filteredTabs?.[(index || 1) - 1]?.id || filteredTabs?.[0]?.id || null;
			setTimeout(() => setCurrFilterTabs(newTab), 0);
		}
	};

	const addFilter = () => {
		const val: Filter = {id: Date.now(), name: 'Logs', def: {level: 'DEBUG', types: []}};
		const allTabs = [...filterTabs, val];
		setFilterTabs(allTabs);
		setCurrFilterTabs(val.id);
	};

	const clicked = (e: React.MouseEvent<HTMLParagraphElement>, tab: Filter) => {
		e.stopPropagation();
		setIsEditing(true);
		setInputVal(tab.name);
		function selectText() {
			const range = document.createRange();
			const selection = window.getSelection();
			if (!e.target) return;
			range.selectNodeContents(e.target as Node);
			selection?.removeAllRanges();
			selection?.addRange(range);
		}
		selectText();
	};

	const editFinished = (e: any, tab: Filter) => {
		setIsEditing(false);
		if (!e.target.textContent) e.target.textContent = inputVal || tab.name;
		tab.name = e.target.textContent;
		localStorage.setItem('filters', JSON.stringify(filterTabs));
		e.target.blur();
	};

	const editCancelled = (e: any, tab: Filter) => {
		setIsEditing(false);
		e.target.textContent = tab.name;
		e.target.blur();
	};

	return (
		<div className={twMerge('flex bg-[#F5F6F8] items-center filter-tabs border-b border-[#DBDCE0] min-h-[36px] max-h-[36px] overflow-x-auto overflow-y-hidden no-scrollbar', isEditing && 'border-[#ebecf0]')}>
			{filterTabs.map((tab: Filter, i: number) => (
				<div className='border-e border-[#DBDCE0]' key={tab.id}>
					<div
						key={tab.id}
						role='button'
						onClick={() => {
							setIsEditing(false);
							setCurrFilterTabs(tab.id);
						}}
						className={twJoin(
							'group flex min-h-[36px] max-w-[200px] max-h-[36px] justify-center leading-[18px] text-[15px] border border-transparent items-center ps-[8px] pe-[8px] p-[10px] border-e w-fit',
							tab.id === currFilterTabs && '!bg-white',
							i === 0 && 'ps-[16px]',
							tab.id === currFilterTabs && isEditing && 'border-b-black border-b-[2px] min-h-[28px] max-h-[28px] !border-[#151515] h-full rounded-[5px]'
						)}>
						<div className={twMerge('flex items-center gap-[8px] relative max-w-full')}>
							<p
								onClick={(e) => tab.id === currFilterTabs && clicked(e, tab)}
								contentEditable={tab.id === currFilterTabs}
								suppressContentEditableWarning
								onKeyDown={(e) => (e.key === 'Enter' ? editFinished(e, tab) : e.key === 'Escape' && editCancelled(e, tab))}
								onBlur={(e) => editFinished(e, tab)}
								className={twMerge(
									'text-[15px] flex-1 overflow-hidden whitespace-nowrap text-ellipsis h-[28px] px-[8px] outline-none items-center border border-transparent flex !justify-start',
									tab.id === currFilterTabs && !isEditing && 'hover:border-gray-200'
								)}>
								{tab.name}
							</p>
							{filterTabs.length > 0 && (
								<X
									role='button'
									className={twJoin('size-[18px] group-hover:visible rounded-[3px]', tab.id !== currFilterTabs && 'invisible group-hover:visible', tab.id === currFilterTabs && isEditing && '!invisible')}
									onClick={() => (tab.id !== currFilterTabs || !isEditing) && deleteFilterTab(tab.id, i)}
								/>
							)}
							{/* {filterTabs.length > 0 && <img src='icons/close.svg' alt='close' className='h-[20px]' role='button' height={10} width={10} onClick={() => deleteFilterTab(tab.id)} />} */}
						</div>
					</div>
				</div>
			))}
			<div className='flex gap-[10px] ms-[6px] items-center rounded-[2px] p-[4px] w-fit sticky right-0 text-[#151515] hover:text-[#151515] hover:bg-[#EBECF0]' role='button' onClick={addFilter}>
				<Plus size={16} />
			</div>
		</div>
	);
};

const EmptyState = ({title, subTitle, className}: {title: string; subTitle?: string; className?: ClassNameValue}) => {
	return (
		<div className={twMerge('flex flex-col m-auto justify-center items-center w-full h-full', className)}>
			<div className='flex flex-col justify-center items-center -translate-y-[70px]'>
				<img className='size-[224px] rounded-full' src='emcie-placeholder.svg' alt='' />
				<h2 className='text-[20px] font-medium font-inter text-[#656565] mt-[30px]'>{title}</h2>
				{subTitle && <p className='text-[15px] font-normal max-w-[378px] font-inter text-[#656565] text-center mt-[10px]'>{subTitle}</p>}
			</div>
		</div>
	);
};

const MessageLogs = ({event, closeLogs, regenerateMessageFn}: {event?: EventInterface | null; closeLogs?: VoidFunction; regenerateMessageFn?: (sessionId: string) => void}): ReactNode => {
	const [filters, setFilters] = useState({});
	const [filterTabs, setFilterTabs] = useLocalStorage<Filter[]>('filters', []);
	const [currFilterTabs, setCurrFilterTabs] = useState<number | null>((filterTabs as Filter[])[0]?.id || null);
	const [logs, setLogs] = useState<Log[] | null>(null);
	const [filteredLogs, setFilteredLogs] = useState<Log[]>([]);
	const messagesRef = useRef<HTMLDivElement | null>(null);
	const [dialog] = useAtom(dialogAtom);

	useEffect(() => {
		if (logs) {
			if (!Object.keys(filters).length) setFilteredLogs(logs);
			else {
				setFilteredLogs(getMessageLogsWithFilters(event?.correlation_id as string, filters as {level: string; types?: string[]; content?: string[]}));
				(setFilterTabs as any)((tabFilters: Filter[]) => {
					if (!tabFilters.length) {
						const filter = {id: Date.now(), def: filters, name: 'Logs'};
						setCurrFilterTabs(filter.id);
						return [filter];
					}
					const tab = tabFilters.find((t) => t.id === currFilterTabs);
					if (!tab) return tabFilters;
					tab.def = filters;
					return [...tabFilters];
				});
			}
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [logs, filters]);

	useEffect(() => {
		if (!event && logs) {
			setLogs(null);
			setFilteredLogs([]);
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [event]);

	useEffect(() => {
		if (!event?.correlation_id) return;
		setLogs(getMessageLogs(event.correlation_id));
	}, [event?.correlation_id]);

	const openLogs = (text: string) => {
		const element = (
			<pre className='group px-[30px] py-[10px] text-wrap text-[#333] relative overflow-auto h-[100%]'>
				<div className='invisble group-hover:visible flex sticky top-[10px] right-[20px] justify-end'>
					<div className='flex justify-end bg-white p-[10px] gap-[20px] rounded-lg'>
						<Tooltip value='Copy' side='top'>
							<Copy onClick={() => copy(text)} size={18} className='cursor-pointer' />
						</Tooltip>
						<Tooltip value='Close' side='top'>
							<X onClick={() => dialog.closeDialog()} size={18} className='cursor-pointer' />
						</Tooltip>
					</div>
				</div>
				<div>{text}</div>
			</pre>
		);
		dialog.openDialog('', element, {height: '90vh', width: '90vw'});
	};

	return (
		<div className={twJoin('w-full h-full overflow-auto flex flex-col justify-start pt-0 pe-0 bg-[#FBFBFB]')}>
			<Header event={event || null} closeLogs={closeLogs} regenerateMessageFn={regenerateMessageFn} className={twJoin(event && logs && !logs?.length && 'bg-white', Object.keys(filters).length ? 'border-[#DBDCE0]' : '')} />
			{event && !!logs?.length && !!filterTabs?.length && <FilterTabs currFilterTabs={currFilterTabs} filterTabs={filterTabs as any} setFilterTabs={setFilterTabs as any} setCurrFilterTabs={setCurrFilterTabs} />}
			{event && (
				<LogFilters
					className={twMerge(!filteredLogs?.length && '', !logs?.length && 'absolute')}
					filterId={currFilterTabs || undefined}
					def={structuredClone((filterTabs as any).find((t: Filter) => currFilterTabs === t.id)?.def || null)}
					applyFn={(types, level, content) => setFilters({types, level, content})}
				/>
			)}
			{!event && <EmptyState title='Feeling curious?' subTitle='Select an agent message for additional actions and information about its generation process.' className='bg-[#f5f6f8]' />}
			{event && logs && !logs?.length && <EmptyState title='Whoopsie!' subTitle="The logs for this message weren't found in cache. Try regenerating it to get fresh logs." className='bg-[#f5f6f8]' />}
			{event && !!logs?.length && !filteredLogs.length && <EmptyState title='No logs current filters' className='bg-[#ebecf0]' />}
			{event && !!filteredLogs.length && (
				<div className='bg-[#EBECF0] p-[14px] pt-0 h-auto overflow-auto flex-1'>
					<div ref={messagesRef} className='rounded-[14px] border-[10px] border-white h-full overflow-auto bg-white fixed-scroll'>
						{filteredLogs.map((log, i) => (
							<div key={i} className={twJoin('flex max-h-[30%] overflow-hidden group relative font-ubuntu-mono rounded-[8px] gap-[5px] px-[20px] p-[14px] border-white border text-[14px] transition-all hover:border-[#EDEDED] hover:bg-[#F5F6F8]')}>
								<div className='absolute hidden group-hover:flex right-[10px] top-[10px] gap-[10px]'>
									<Tooltip value='Copy' side='top'>
										<Copy size={18} onClick={() => copy(log?.message || '')} className='cursor-pointer' />
									</Tooltip>
									<Tooltip value='Full screen' side='top'>
										<Fullscreen size={20} className='cursor-pointer' onClick={() => openLogs(log?.message || '')} />
									</Tooltip>
								</div>
								<pre className={clsx('max-w-[-webkit-fill-available] pe-[10px] text-wrap [mask-image:linear-gradient(to_bottom,white_60px,_transparent)]')}>
									{' '}
									{log?.level ? `[${log.level}]` : ''}
									{log?.message}
								</pre>
							</div>
						))}
					</div>
				</div>
			)}
		</div>
	);
};

export default MessageLogs;
