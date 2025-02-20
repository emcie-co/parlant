import {memo, ReactNode, useEffect, useRef, useState} from 'react';
import {Button} from '../ui/button';
import {Checkbox} from '../ui/checkbox';
import {Input} from '../ui/input';
import {Dialog, DialogClose, DialogContent, DialogDescription, DialogPortal, DialogTitle, DialogTrigger} from '../ui/dialog';
import {ClassNameValue, twMerge} from 'tailwind-merge';
import {ListFilter, Plus, X} from 'lucide-react';
import {getDistanceToRight} from '@/utils/methods';
import Tooltip from '../ui/custom/tooltip';

export type Type = 'GuidelineProposer' | 'MessageEventComposer' | 'ToolCaller';
export type Level = 'WARNING' | 'INFO' | 'DEBUG';

const ALL_TYPES: Type[] = ['GuidelineProposer', 'ToolCaller', 'MessageEventComposer'];
const ALL_LEVELS: Level[] = ['WARNING', 'INFO', 'DEBUG'];

const typeLabels: Record<Type, string> = {
	GuidelineProposer: 'Guideline Proposer',
	MessageEventComposer: 'Message Event Composer',
	ToolCaller: 'Tool Caller',
};

const AddFilterChip = ({className}: {className?: ClassNameValue}) => {
	return (
		<div className={twMerge('group cursor-pointer bg-[#F5F6F8] hover:bg-[#EBECF0] h-[30px] rounded-[3px] flex items-center w-full', className)}>
			<div className='flex items-center rounded-[3px] h-[calc(100%-4px)] w-[calc(100%-4px)] py-[5px] px-[8px] pe-[6px] gap-[8px]'>
				<Plus role='button' className='min-w-[18px] size-[18px] rounded-[3px]' />
				<p className='text-nowrap font-normal text-[14px]'>Add Custom Filter</p>
			</div>
		</div>
	);
};

const FilterDialogContent = ({contentChanged, defaultValue}: {contentChanged: (text: string) => void; defaultValue?: string}) => {
	const [inputVal, setInputVal] = useState(defaultValue || '');

	const onApplyClick = () => {
		const trimmed = inputVal.trim();
		if (trimmed) contentChanged(inputVal);
	};

	return (
		<div className='px-[39px] py-[42px] flex flex-col gap-[22px]'>
			<h2 className='text-[20px] font-normal'>Filter by content</h2>
			<div className='border rounded-[5px] h-[38px] flex items-center bg-[#FBFBFB] hover:bg-[#F5F6F8] focus-within:!bg-white'>
				<Input value={inputVal} onChange={(e) => setInputVal(e.target.value)} name='filter' className='h-[36px] !ring-0 !ring-offset-0 border-none text-[16px] bg-[#FBFBFB] hover:bg-[#F5F6F8] focus:!bg-white' />
			</div>
			<div className='buttons flex items-center gap-[16px] justify-end text-[16px] font-normal font-ubuntu-sans'>
				<DialogClose className='h-[38px] w-[84px] !bg-white text-[#656565] hover:text-[#151515] rounded-[5px] border'>Cancel</DialogClose>
				<DialogClose onClick={onApplyClick} className='bg-[#151515] text-white h-[38px] w-[79px] hover:bg-black rounded-[5px]'>
					Apply
				</DialogClose>
			</div>
		</div>
	);
};

const FilterDialog = ({contentChanged, content, children, className}: {contentChanged: (text: string) => void; content?: string; children?: ReactNode; className?: ClassNameValue}) => {
	return (
		<Dialog>
			<DialogTrigger className='w-full'>
				{children || <AddFilterChip className={className} />}

				{/* <div className='group border rounded-[3px] h-[24px] flex items-center bg-[#FBFBFB] hover:bg-[#F5F6F8]'>
					<p className='ps-[10px] text-[12px] capitalize'>Content:</p>
					<Input readOnly className='h-[22px] !ring-0 !ring-offset-0 border-none text-[12px] bg-[#FBFBFB] hover:bg-[#F5F6F8]' value={content?.join(';') || ''} />
				</div> */}
			</DialogTrigger>
			<DialogPortal aria-hidden={false}>
				<DialogContent aria-hidden={false} className='p-0 [&>button]:hidden'>
					<DialogTitle className='hidden'>Filter by content</DialogTitle>
					<DialogDescription className='hidden'>Filter by content</DialogDescription>
					<FilterDialogContent contentChanged={contentChanged} defaultValue={content || ''} />
				</DialogContent>
			</DialogPortal>
		</Dialog>
	);
};

const LogFilters = ({applyFn, def, filterId, className}: {applyFn: (types: string[], level: string, content: string[]) => void; filterId?: number; def?: {level?: Level; types?: Type[]; content?: string[]} | null; className?: ClassNameValue}) => {
	const [sources, setSources] = useState(structuredClone(def?.types || []));
	const [contentConditions, setContentConditions] = useState(structuredClone(def?.content || []));
	const [level, setLevel] = useState<Level>(def?.level || ALL_LEVELS[ALL_LEVELS.length - 1]);

	useEffect(() => {
		if (filterId) {
			const types = structuredClone(def?.types || ALL_TYPES);
			const level = def?.level || ALL_LEVELS[ALL_LEVELS.length - 1];
			const content = def?.content || [];
			setSources(types);
			setLevel(level);
			setContentConditions(content);
			applyFn(types, level, content);
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [filterId]);

	useEffect(() => {
		setSources(def?.types || []);
		setLevel(def?.level || ALL_LEVELS[ALL_LEVELS.length - 1]);
		setContentConditions(def?.content || []);
	}, [def]);

	const changeSource = (type: Type, value: boolean, cb?: (sources: Type[], level: Level, contentConditions: string[]) => void) => {
		setSources((val) => {
			if (value) val.push(type);
			else val = val.filter((item) => item !== type);
			const vals = [...new Set(val)];
			cb?.(vals, level, contentConditions);
			return vals;
		});
	};

	const TypeChip = ({type}: {type: Type}) => {
		return (
			<div key={type} className='group cursor-pointer bg-[#F5F6F8] h-[30px] flex items-center gap-[8px] pt-[6px] pb-[5px] ps-[14px] rounded-[5px] pe-[8px] hover:bg-white'>
				<p className='text-nowrap font-normal text-[14px]'>{typeLabels[type]}</p>
				<X role='button' className='invisible size-[18px] group-hover:visible rounded-[3px]' onClick={() => changeSource(type, false, applyFn)} />
			</div>
		);
	};

	const CondChip = ({
		text,
		index,
		apply,
		deleted,
		wrapperClassName,
		className,
		deleteButtonClassName,
	}: {
		text: string;
		index: number;
		apply?: boolean;
		deleted?: (content: string[]) => void;
		className?: ClassNameValue;
		wrapperClassName?: ClassNameValue;
		deleteButtonClassName?: ClassNameValue;
	}) => {
		return (
			<Tooltip value={text} side='top' delayDuration={1000}>
				<div key={text} className={twMerge('group px-[2px] max-w-[320px] cursor-pointer bg-[#F5F6F8] hover:bg-white border-[#A9A9A9] hover:border-[#D7D7D7] border h-[30px] rounded-[5px] flex justify-center items-center w-fit', wrapperClassName)}>
					<div
						className={twMerge('flex items-center justify-center max-w-full rounded-[3px] h-[calc(100%-4px)] py-[5px] ps-[14px] pe-[6px] gap-[8px] bg-white group-hover:bg-[#F5F6F8] border border-[#D7D7D7] group-hover:border-[#A9A9A9]', className)}>
						<p className='text-nowrap max-w-full overflow-hidden text-ellipsis font-normal text-[14px]'>{text}</p>
						<X
							role='button'
							className={twMerge('invisible min-w-[18px] size-[18px] group-hover:visible rounded-[3px]', deleteButtonClassName)}
							onClick={(e) => {
								e.stopPropagation();
								const content = contentConditions?.filter((_, i) => i !== index);
								if (apply) {
									setContentConditions(content);
									applyFn(sources, level, content);
								}
								deleted?.(content);
							}}
						/>
					</div>
				</div>
			</Tooltip>
		);
	};

	const DropDownFilter = () => {
		const [dropdownOpen, setDropdownOpen] = useState(false);
		const [sources, setSources] = useState<Type[]>(def?.types || []);
		const [content, setContent] = useState<string[]>(def?.content || []);
		const wrapperRef = useRef<HTMLDivElement>(null);
		const [usePopupToLeft, setUsePopupToLeft] = useState(false);

		const changeSource = (type: Type, value: boolean) => {
			setSources((val) => {
				if (value) val.push(type);
				else val = val.filter((item) => item !== type);
				const vals = [...new Set(val)];
				return vals;
			});
		};

		useEffect(() => {
			if (!dropdownOpen) {
				setSources(structuredClone(def?.types || []));
				setContent(structuredClone(def?.content || []));
			}
		}, [dropdownOpen]);

		useEffect(() => {
			if (wrapperRef?.current) {
				if (getDistanceToRight(wrapperRef.current) < 218) setUsePopupToLeft(true);
				else setUsePopupToLeft(false);
			}
		}, [wrapperRef?.current?.scrollWidth, dropdownOpen]);

		return (
			<div className='wrapper relative' ref={wrapperRef}>
				<div>
					{!def?.types?.length ? (
						<div onClick={() => setDropdownOpen(true)} role='button' className={twMerge('flex bg-[#EDEDED] hover:bg-[#F5F6F8] rounded-[5px] items-center gap-[6px] h-[30px] px-[14px]', dropdownOpen && 'bg-[#CDCDCD]')}>
							<ListFilter className='[stroke-width:2px] size-[16px]' />
							<p className='text-[14px] font-medium'>Filters</p>
						</div>
					) : (
						<Button onClick={() => setDropdownOpen(true)} variant='ghost' className={twMerge('flex bg-[#EDEDED] hover:bg-[#F5F6F8] items-center gap-[6px] w-[32px] h-[30px] p-0', dropdownOpen && 'bg-[#CDCDCD]')}>
							<img src='icons/filter.svg' alt='' />
						</Button>
					)}
				</div>
				<div className={twMerge('hidden border rounded-[7px] absolute top-[38px] left-0 w-[218px] z-10 bg-white', dropdownOpen && 'block', usePopupToLeft ? 'right-0 left-[unset]' : '')}>
					<div className='flex justify-between items-center'>
						<div className='flex items-center gap-[6px] h-[35px] px-[14px]'>
							{/* <ListFilter className='[stroke-width:2px] size-[16px]' /> */}
							<p className='text-[14px] font-normal'>Filter</p>
						</div>
						<div role='button' onClick={() => setDropdownOpen(false)} className='flex h-[24px] w-[24px] items-center me-[2px] justify-center'>
							<img src='icons/close.svg' alt='close' />
						</div>
					</div>
					<hr className='bg-[#EBECF0]' />
					<div className='flex flex-col gap-[4px] mt-[9px] pb-[11px] px-[8px]'>
						{ALL_TYPES.map((type) => (
							<div key={type} className={twMerge('flex items-center py-[4px] ps-[6px] space-x-2 hover:bg-[#F5F6F8]', sources.includes(type) && 'bg-[#EBECF0]')}>
								<Checkbox id={type} defaultChecked={def?.types?.includes(type)} className='border-black rounded-[2px] !bg-white' onCheckedChange={(isChecked) => changeSource(type, !!isChecked)} />
								<label className='text-[12px] font-normal w-full cursor-pointer' htmlFor={type}>
									{typeLabels[type]}
								</label>
							</div>
						))}
					</div>
					<hr className='bg-[#EBECF0]' />
					<div className={twMerge('inputs flex flex-wrap gap-[6px] px-[14px] pb-[14px] pt-[11px]', !content?.length && 'h-0 p-0')}>
						{content?.map((item, i) => (
							<FilterDialog
								key={item}
								content={item}
								contentChanged={(inputVal) => {
									setContent((c) => {
										c[i] = inputVal;
										return [...c];
									});
								}}>
								<CondChip
									text={item}
									index={i}
									apply={false}
									deleted={(content) => setContent(content)}
									wrapperClassName='w-full !border-0 bg-[#F5F6F8] hover:bg-[#EBECF0]'
									className='justify-between !border-0 bg-[#F5F6F8] group-hover:bg-[#EBECF0]'
									deleteButtonClassName='visible'
								/>
							</FilterDialog>
						))}
					</div>
					{!!content?.length && <hr className='bg-[#EBECF0] w-full' />}
					<div className='px-[14px] h-[54px] flex items-center'>
						<FilterDialog contentChanged={(inputVal) => setContent((val) => [...val, inputVal])} />
					</div>
					<hr className='bg-[#EBECF0]' />
					<div className='buttons flex items-center h-[47px] p-[6px]'>
						<Button onClick={() => applyFn([], 'DEBUG', [])} variant='ghost' className='flex-1 text-[12px] hover:text-[#151515] hover:bg-transparent font-normal text-[#656565] h-[35px] w-[95px]'>
							Clear all
						</Button>
						<Button
							variant='ghost'
							onClick={() => {
								applyFn(sources, level, content);
								setDropdownOpen(false);
							}}
							className='flex-1 text-[12px] font-normal !text-white bg-[#151515] h-[35px] w-[95px] hover:bg-black'>
							Apply
						</Button>
					</div>
				</div>
			</div>
		);
	};

	return (
		<div className={twMerge('flex justify-between py-[10px] pe-[10px] ps-[14px] bg-[#EBECF0] min-h-fit h-[58px]', (!!def?.types?.length || !!def?.content?.length) && 'h-[50px]', className)}>
			<div className='filters-button flex items-center gap-[8px] flex-wrap'>
				{!!def?.types?.length && def.types.map((type) => <TypeChip key={type} type={type} />)}
				{def?.content?.map((c: string, index: number) => (
					<Dialog key={c}>
						<DialogTrigger>
							<CondChip key={c} text={c} index={index} apply={true} />
						</DialogTrigger>
						<DialogPortal>
							<DialogContent className='p-0'>
								<DialogTitle hidden>Filter by content</DialogTitle>
								<DialogDescription hidden>Filter by content</DialogDescription>
								<FilterDialogContent
									defaultValue={c}
									contentChanged={(text) => {
										const updatedContent = contentConditions.map((item, i) => (i === index ? text : item));
										applyFn(sources, level, updatedContent);
									}}
								/>
							</DialogContent>
						</DialogPortal>
					</Dialog>
				))}
				<DropDownFilter />
			</div>
		</div>
	);
};

export default memo(LogFilters);
