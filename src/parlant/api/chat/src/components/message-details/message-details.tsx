/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable react-hooks/exhaustive-deps */
import {EventInterface, Log} from '@/utils/interfaces';
import React, {ReactNode, useEffect, useRef, useState} from 'react';
import {getMessageLogs, getMessageLogsWithFilters} from '@/utils/logs';
import {twJoin, twMerge} from 'tailwind-merge';
import clsx from 'clsx';
import {useLocalStorage} from '@/hooks/useLocalStorage';
import LogFilters, {Level, Type} from '../log-filters/log-filters';
import MessageFragments from '../message-fragments/message-fragments';
import EmptyState from './empty-state';
import FilterTabs from './filter-tabs';
import MessageDetailsHeader from './message-details-header';
import {ResizableHandle, ResizablePanel, ResizablePanelGroup} from '../ui/resizable';
import {ImperativePanelHandle} from 'react-resizable-panels';
import Tooltip from '../ui/custom/tooltip';
import {copy} from '@/lib/utils';
import MessageLogs from './message-logs';

interface DefInterface {
	level?: Level;
	types?: Type[];
	content?: string[];
}

interface Filter {
	id: number;
	name: string;
	def: DefInterface | null;
}

const MessageError = ({event}: {event: EventInterface}) => {
	return (
		<div className='h-full group p-[20px] bg-[#ebecf0] text-[13px] text-[#ef5350] z-10'>
			<pre className={clsx('p-[10px] max-w-[-webkit-fill-available] pe-[10px] text-wrap break-words bg-white rounded-[8px] h-full overflow-auto  group relative')}>
				<div className='sticky h-0 hidden z-10 group-hover:block [direction:rtl] top-[10px] right-[10px] gap-[10px]'>
					<Tooltip value='Copy' side='top'>
						<img src='icons/copy.svg' sizes='18' alt='' onClick={() => copy(event?.error || '')} className='cursor-pointer' />
					</Tooltip>
				</div>
				{event?.error}
			</pre>
		</div>
	);
};

const MessageDetails = ({
	event,
	closeLogs,
	regenerateMessageFn,
	resendMessageFn,
}: {
	event?: EventInterface | null;
	closeLogs?: VoidFunction;
	regenerateMessageFn?: (sessionId: string) => void;
	resendMessageFn?: (sessionId: string) => void;
}): ReactNode => {
	const [filters, setFilters] = useState({});
	const [filterTabs, setFilterTabs] = useLocalStorage<Filter[]>('filters', []);
	const [currFilterTabs, setCurrFilterTabs] = useState<number | null>((filterTabs as Filter[])[0]?.id || null);
	const [logs, setLogs] = useState<Log[] | null>(null);
	const [filteredLogs, setFilteredLogs] = useState<Log[]>([]);
	const messagesRef = useRef<HTMLDivElement | null>(null);
	const resizableRef = useRef<ImperativePanelHandle | null>(null);

	useEffect(() => {
		if (event?.id) resizableRef.current?.resize(50);
	}, [event?.id]);

	useEffect(() => {
		const hasFilters = Object.keys(filters).length;
		if (logs) {
			if (!hasFilters) setFilteredLogs(logs);
			else {
				setFilteredLogs(getMessageLogsWithFilters(event?.correlation_id as string, filters as {level: string; types?: string[]; content?: string[]}));
				(setFilterTabs as React.Dispatch<React.SetStateAction<Filter[]>>)((tabFilters: Filter[]) => {
					if (!tabFilters.length && hasFilters) {
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
	}, [logs, filters]);

	useEffect(() => {
		if (!event && logs) {
			setLogs(null);
			setFilteredLogs([]);
		}
	}, [event]);

	useEffect(() => {
		if (!event?.correlation_id) return;
		setLogs(getMessageLogs(event.correlation_id));
	}, [event?.correlation_id]);

	const deleteFilterTab = (id: number) => {
		const filterIndex = (filterTabs as Filter[]).findIndex((t) => t.id === id);
		if (filterIndex === -1) return;
		const filteredTabs = (filterTabs as Filter[]).filter((t) => t.id !== id);
		setFilterTabs(filteredTabs);

		if (currFilterTabs === id) {
			const newTab = filteredTabs?.[(filterIndex || 1) - 1]?.id || filteredTabs?.[0]?.id || null;
			setCurrFilterTabs(newTab);
		}
		if (!filteredTabs.length) setFilters({});
	};

	const shouldRenderTabs = event && !!logs?.length && !!filterTabs?.length;
	const fragmentEntries = Object.entries(event?.data?.fragments || {}).map(([id, value]) => ({id, value}));
	const isError = event?.serverStatus === 'error';

	return (
		<div className={twJoin('w-full h-full overflow-auto flex flex-col justify-start pt-0 pe-0 bg-[#FBFBFB]')}>
			<MessageDetailsHeader
				event={event || null}
				closeLogs={closeLogs}
				resendMessageFn={resendMessageFn}
				regenerateMessageFn={regenerateMessageFn}
				className={twJoin('shadow-main h-[60px] min-h-[60px]', Object.keys(filters).length ? 'border-[#F3F5F9]' : '')}
			/>
			<ResizablePanelGroup direction='vertical' className={twJoin('w-full h-full overflow-auto flex flex-col justify-start pt-0 pe-0 bg-[#FBFBFB]')}>
				<ResizablePanel ref={resizableRef} minSize={0} maxSize={isError ? 99 : 0} defaultSize={isError ? 50 : 0}>
					{isError && <MessageError event={event} />}
				</ResizablePanel>
				<ResizableHandle withHandle className={twJoin(!isError && 'hidden')} />
				<ResizablePanel minSize={isError ? 0 : 100} maxSize={isError ? 99 : 100} defaultSize={isError ? 50 : 100} className='flex flex-col bg-white'>
					{!!fragmentEntries.length && <MessageFragments fragments={fragmentEntries} />}
					<div className='flex justify-between items-center min-h-[58px] h-[58px] p-[10px] pb-[4px] pe-0'>
						<div className='ps-[14px] text-[#282828]'>Logs</div>
						{!shouldRenderTabs && (
							<LogFilters
								showDropdown
								filterId={currFilterTabs || undefined}
								def={structuredClone((filterTabs as Filter[]).find((t: Filter) => currFilterTabs === t.id)?.def || null)}
								applyFn={(types, level, content) => setFilters({types, level, content})}
							/>
						)}
					</div>
					{shouldRenderTabs && <FilterTabs currFilterTabs={currFilterTabs} filterTabs={filterTabs as Filter[]} setFilterTabs={setFilterTabs as any} setCurrFilterTabs={setCurrFilterTabs} />}
					{event && !!logs?.length && shouldRenderTabs && (
						<LogFilters
							showTags
							showDropdown
							deleteFilterTab={deleteFilterTab}
							className={twMerge(!filteredLogs?.length && '', !logs?.length && 'absolute')}
							filterId={currFilterTabs || undefined}
							def={structuredClone((filterTabs as Filter[]).find((t: Filter) => currFilterTabs === t.id)?.def || null)}
							applyFn={(types, level, content) => setFilters({types, level, content})}
						/>
					)}
					{!event && <EmptyState title='Feeling curious?' subTitle='Select a message for additional actions and information about its process.' wrapperClassName='bg-[#f5f6f8]' />}
					{event && logs && !logs?.length && (
						<EmptyState title='Whoopsie!' subTitle="The logs for this message weren't found in cache. Try regenerating it to get fresh logs." wrapperClassName='bg-[#f5f6f8]' className={twJoin(isError && 'translate-y-[0px]')} />
					)}
					{event && !!logs?.length && !filteredLogs.length && <EmptyState title='No logs for the current filters' wrapperClassName='bg-[#ebecf0]' className={twJoin(isError && 'translate-y-[0px]')} />}
					{event && !!filteredLogs.length && (
						<div className='ps-[10px] overflow-auto h-[-webkit-fill-available]'>
							<MessageLogs messagesRef={messagesRef} filteredLogs={filteredLogs} />
						</div>
					)}
				</ResizablePanel>
			</ResizablePanelGroup>
		</div>
	);
};

export default MessageDetails;
