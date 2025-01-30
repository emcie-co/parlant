import {EventInterface, Log} from '@/utils/interfaces';
import {Plus, X} from 'lucide-react';
import {ReactNode, useEffect, useRef, useState} from 'react';
import {getMessageLogs, getMessageLogsWithFilters} from '@/utils/logs';
import {ClassNameValue, twJoin, twMerge} from 'tailwind-merge';
import clsx from 'clsx';
import HeaderWrapper from '../header-wrapper/header-wrapper';
import {useLocalStorage} from '@/hooks/useLocalStorage';
import LogFilters from '../log-filters/log-filters';
import {useAtom} from 'jotai';
import {sessionAtom} from '@/store';
import Markdown from '../markdown/markdown';
import CopyText from '../ui/custom/copy-text';

interface Filter {
	id: number;
	name: string;
	def: {level?: string; types?: string[]} | null;
}

interface FilterTabsFilterProps {
	filterTabs: Filter[];
	setCurrFilterTabs: React.Dispatch<React.SetStateAction<number | null>>;
	setFilterTabs: React.Dispatch<React.SetStateAction<Filter[] | (() => Filter[])>>;
	currFilterTabs: number | null;
}

const Header = ({event, regenerateMessageFn, closeLogs}: {event: EventInterface | null; regenerateMessageFn?: (messageId: string) => void; closeLogs?: VoidFunction}) => {
	const [session] = useAtom(sessionAtom);
	return (
		<HeaderWrapper className={twMerge('static bg-[#FBFBFB]', !event && 'border-transparent')}>
			{event && (
				<div className={twMerge('flex items-center justify-between w-full pe-[20px]')}>
					<div className='flex items-center gap-[12px]'>
						<div className='group flex rounded-[5px] ms-[14px] items-center gap-[7px] hover:bg-[#F5F6F8] py-[13px] px-[10px]' role='button' onClick={() => regenerateMessageFn?.(session?.id as string)}>
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
	const deleteFilterTab = (id: number) => {
		const filteredTabs = filterTabs.filter((t) => t.id !== id);
		setFilterTabs(filteredTabs);

		if (currFilterTabs === id) {
			const firstTab = filteredTabs[0]?.id || null;
			setTimeout(() => setCurrFilterTabs(firstTab), 0);
		}
	};

	const addFilter = () => {
		const val: Filter = {id: Date.now(), name: 'untitled', def: {level: 'DEBUG', types: []}};
		const allTabs = [...filterTabs, val];
		setFilterTabs(allTabs);
		setCurrFilterTabs(val.id);
	};

	const clicked = (e: MouseEvent, tab: Filter) => {
		e.stopPropagation();
		setIsEditing(true);
		setInputVal(tab.name);
		function selectText() {
			const range = document.createRange();
			const selection = window.getSelection();
			if (!e.target) return;
			range.selectNodeContents(e.target as any);
			selection?.removeAllRanges();
			selection?.addRange(range);
		}
		selectText();
	};

	const editFinished = (e, tab: Filter) => {
		setIsEditing(false);
		if (!e.target.textContent) e.target.textContent = inputVal || tab.name;
		tab.name = e.target.textContent;
		localStorage.setItem('filters', JSON.stringify(filterTabs));
		e.target.blur();
	};

	const editCancelled = (e, tab: Filter) => {
		setIsEditing(false);
		e.target.textContent = tab.name;
		e.target.blur();
	};

	return (
		<div className='flex bg-[#F5F6F8] items-center filter-tabs border-b min-h-[36px] max-h-[36px] overflow-x-auto overflow-y-hidden no-scrollbar'>
			{filterTabs.map((tab: Filter, i: number) => (
				<div
					key={tab.id}
					role='button'
					onClick={() => {
						setIsEditing(false);
						setCurrFilterTabs(tab.id);
					}}
					className={twJoin(
						'group flex min-h-[36px] max-h-[36px] justify-center leading-[18px] text-[15px] border border-transparent items-center ps-[8px] pe-[8px] p-[10px] border-e w-fit',
						tab.id === currFilterTabs && '!bg-white',
						i === 0 && 'ps-[16px]',
						tab.id === currFilterTabs && isEditing && 'border-b-black ms-[3px] min-h-[28px] max-h-[28px] -me-[3px] !border-[#151515] h-full rounded-[2px]'
					)}>
					<div className={twMerge('flex items-center gap-[8px] relative')}>
						<p
							onClick={(e) => tab.id === currFilterTabs && clicked(e, tab)}
							contentEditable={tab.id === currFilterTabs}
							suppressContentEditableWarning
							onKeyDown={(e) => (e.key === 'Enter' ? editFinished(e, tab) : e.key === 'Escape' && editCancelled(e, tab))}
							onBlur={(e) => editFinished(e, tab)}
							className={twMerge('text-[15px] h-[28px] px-[8px] outline-none flex items-center border border-transparent', tab.id === currFilterTabs && !isEditing && 'hover:border-gray-200')}>
							{tab.name}
						</p>
						{filterTabs.length > 0 && (
							<X
								role='button'
								className={twJoin('size-[18px] group-hover:visible rounded-[3px]', tab.id !== currFilterTabs && 'invisible group-hover:visible', tab.id === currFilterTabs && isEditing && '!invisible')}
								onClick={() => (tab.id !== currFilterTabs || !isEditing) && deleteFilterTab(tab.id)}
							/>
						)}
						{/* {filterTabs.length > 0 && <img src='icons/close.svg' alt='close' className='h-[20px]' role='button' height={10} width={10} onClick={() => deleteFilterTab(tab.id)} />} */}
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
			<img className='size-[224px] rounded-full' src='emcie-placeholder.svg' alt='' />
			<h2 className='text-[20px] font-medium font-inter text-[#656565] mt-[30px]'>{title}</h2>
			{subTitle && <p className='text-[15px] font-normal max-w-[378px] font-inter text-[#656565] text-center mt-[10px]'>{subTitle}</p>}
		</div>
	);
};

const MessageLogs = ({event, closeLogs, regenerateMessageFn}: {event?: EventInterface | null; closeLogs?: VoidFunction; regenerateMessageFn?: (sessionId: string) => void}): ReactNode => {
	const [filters, setFilters] = useState({});
	const [filterTabs, setFilterTabs] = useLocalStorage<any>('filters', []);
	const [currFilterTabs, setCurrFilterTabs] = useState<number | null>((filterTabs as Filter[])[0]?.id || null);
	const [logs, setLogs] = useState<Log[]>([]);
	const [filteredLogs, setFilteredLogs] = useState<Log[]>([]);
	const messagesRef = useRef<HTMLDivElement | null>(null);

	// useEffect(() => {
	// 	if (messagesRef.current) messagesRef.current.scrollTo(0, messagesRef.current.scrollHeight);
	// }, [messagesRef?.current]);

	useEffect(() => {
		if (logs) {
			if (!Object.keys(filters).length) setFilteredLogs(logs);
			else {
				setFilteredLogs(getMessageLogsWithFilters(event?.correlation_id as string, filters as {level: string; types?: string[]; content?: string[]}));
				setFilterTabs((tabFilters: Filter[]) => {
					if (!tabFilters.length) {
						const filter = {id: Date.now(), def: filters, name: 'untitled'};
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
		if (!event && logs.length) {
			setLogs([]);
			setFilteredLogs([]);
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [event]);

	useEffect(() => {
		if (!event?.correlation_id) return;
		setLogs(getMessageLogs(event.correlation_id));
	}, [event?.correlation_id]);

	return (
		<div className={twJoin('w-full h-full overflow-auto flex flex-col justify-start pt-0 pe-0 bg-[#FBFBFB]')}>
			<Header event={event || null} closeLogs={closeLogs} regenerateMessageFn={regenerateMessageFn} />
			{event && !!logs.length && !!filterTabs?.length && <FilterTabs currFilterTabs={currFilterTabs} filterTabs={filterTabs} setFilterTabs={setFilterTabs} setCurrFilterTabs={setCurrFilterTabs} />}
			{event && (
				<LogFilters
					className={event && !filteredLogs?.length && 'bg-[#f5f6f8]'}
					filterId={currFilterTabs || undefined}
					def={structuredClone(filterTabs.find((t: Filter) => currFilterTabs === t.id)?.def || null)}
					applyFn={(types, level, content) => setFilters({types, level, content})}
				/>
			)}
			{!event && <EmptyState title='No message has been selected' subTitle='Please select one of the messages so we can give you more information' />}
			{event && !logs.length && <EmptyState title='No Logs Found' subTitle='Please select a different message in the session.' className='bg-[#f5f6f8]' />}
			{event && !!logs.length && !filteredLogs.length && <EmptyState title='No Data' className='bg-[#f5f6f8]' />}
			{event && !!filteredLogs.length && (
				<div className='bg-[#EBECF0] p-[14px] pt-0 h-auto overflow-auto flex-1'>
					<div ref={messagesRef} className='rounded-[14px] border-[10px] border-white h-full overflow-auto bg-white fixed-scroll'>
						{filteredLogs.map((log, i) => (
							<div key={i} className={twJoin('flex rounded-[8px] items-center gap-[5px] px-[20px] p-[5px] border-white border transition-all hover:border-[#EDEDED] hover:bg-[#F5F6F8]')}>
								<Markdown className={clsx('max-w-[-webkit-fill-available] pe-[10px]')}>{log?.message}</Markdown>
							</div>
						))}
					</div>
				</div>
			)}
		</div>
	);
};

export default MessageLogs;
