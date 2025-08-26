import {memo, ReactNode, useEffect, useRef, useState} from 'react';
import {Button} from '../ui/button';
import {Checkbox} from '../ui/checkbox';
import {Input} from '../ui/input';
import {Dialog, DialogClose, DialogContent, DialogDescription, DialogPortal, DialogTitle, DialogTrigger} from '../ui/dialog';
import {ClassNameValue, twMerge} from 'tailwind-merge';
import {X} from 'lucide-react';
import {getDistanceToRight} from '@/utils/methods';
import Tooltip from '../ui/custom/tooltip';
import {Select, SelectContent, SelectGroup, SelectItem, SelectTrigger, SelectValue} from '../ui/select';

export type Type = 'GuidelineMatcher' | 'MessageEventComposer' | 'ToolCaller';
export type Level = 'CRITICAL' | 'ERROR' | 'WARNING' | 'INFO' | 'DEBUG' | 'TRACE';

const ALL_TYPES: Type[] = ['GuidelineMatcher', 'ToolCaller', 'MessageEventComposer'];
const ALL_LEVELS: Level[] = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'TRACE'];

const typeOptions: {[key in Type]: {label: string; icon: string; color: string}} = {
	GuidelineMatcher: {
		label: 'Guideline Matcher',
		icon: 'icons/filters/guideline-matcher-color.svg',
		color: '#419480',
	},
	MessageEventComposer: {
		label: 'Message Event Composer',
		icon: 'icons/filters/message-composer-color.svg',
		color: '#7E3A89',
	},
	ToolCaller: {
		label: 'Tool Caller',
		icon: 'icons/filters/tool-caller-color.svg',
		color: '#CB7714',
	},
};

const AddFilterChip = ({className}: {className?: ClassNameValue}) => {
	return (
		<div className={twMerge('group cursor-pointer bg-white border-[#eeeeee] hover:bg-[#F3F5F9] hover:border-[#E4E6EA] border h-[30px] rounded-[6px] flex items-center w-full shadow-main', className)}>
			<div className='flex items-center justify-center rounded-[3px] leading-[16px] h-[calc(100%-4px)] w-[calc(100%-4px)] py-[5px] px-[8px] pe-[6px]'>
				{/* <p className='me-[5px] text-[14px]'>+</p> */}
				<img src='icons/text.svg' alt='' className='me-[5px]' />
				<p className='text-nowrap font-normal text-[14px]'>Add Content Filter</p>
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
			<div className='buttons flex items-center gap-[16px] justify-end text-[16px] font-normal font-inter'>
				<DialogClose className='h-[38px] w-[84px] !bg-white text-[#656565] hover:text-[#151515] rounded-[5px] border'>Cancel</DialogClose>
				<DialogClose onClick={onApplyClick} className='bg-green-main text-white h-[38px] w-[79px] hover:bg-green-hover rounded-[5px]'>
					Apply
				</DialogClose>
			</div>
		</div>
	);
};

const FilterDialog = ({contentChanged, content, children, className}: {contentChanged: (text: string) => void; content?: string; children?: ReactNode; className?: ClassNameValue}) => {
	return (
		<Dialog>
			<DialogTrigger className='w-full'>{children || <AddFilterChip className={className} />}</DialogTrigger>
			<DialogPortal aria-hidden={false}>
				<DialogContent aria-hidden={false} className='p-0 [&>button]:hidden z-[99]'>
					<DialogTitle className='hidden'>Filter by content</DialogTitle>
					<DialogDescription className='hidden'>Filter by content</DialogDescription>
					<FilterDialogContent contentChanged={contentChanged} defaultValue={content || ''} />
				</DialogContent>
			</DialogPortal>
		</Dialog>
	);
};

const LogFilters = ({
	applyFn,
	def,
	filterId,
	className,
	showDropdown,
	showTags,
	deleteFilterTab,
}: {
	applyFn: (types: string[], level: string, content: string[]) => void;
	filterId?: number;
	def?: {level?: Level; types?: Type[]; content?: string[]} | null;
	className?: ClassNameValue;
	showDropdown?: boolean;
	showTags?: boolean;
	deleteFilterTab?: (filterId: number | undefined) => void;
}) => {
	const [sources, setSources] = useState(structuredClone(def?.types || []));
	const [contentConditions, setContentConditions] = useState(structuredClone(def?.content || []));
	const [level, setLevel] = useState<Level>(def?.level || ALL_LEVELS[ALL_LEVELS.length - 1]);
	const [prevTabId, setPrevTabId] = useState<number | undefined>();

	useEffect(() => {
		if (!showTags) return;
		if (filterId && filterId !== prevTabId) {
			const types = structuredClone(def?.types || ALL_TYPES);
			const level = def?.level || ALL_LEVELS[ALL_LEVELS.length - 1];
			const content = def?.content || [];
			setSources(types);
			setLevel(level);
			setContentConditions(content);
			applyFn(types, level, content);
			setPrevTabId(filterId);
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [filterId]);

	useEffect(() => {
		setSources(def?.types || []);
		setLevel(def?.level || ALL_LEVELS[ALL_LEVELS.length - 1]);
		setContentConditions(def?.content || []);
	}, [def]);

	// const changeSource = (type: Type, value: boolean, cb?: (sources: Type[], level: Level, contentConditions: string[]) => void) => {
	// 	setSources((val) => {
	// 		if (value) val.push(type);
	// 		else val = val.filter((item) => item !== type);
	// 		const vals = [...new Set(val)];
	// 		cb?.(vals, level, contentConditions);
	// 		return vals;
	// 	});
	// };

	const TypeChip = ({type, className}: {type: Type; className?: ClassNameValue}) => {
		return (
			<div key={type} className={twMerge('group border cursor-default border-[#EEEEEE] h-[30px] flex items-center gap-[8px] pt-[6px] pb-[5px] ps-[6px] rounded-[5px] pe-[6px] hover:bg-white', className)}>
				<img src={typeOptions[type].icon} alt={type} />
				<p className='text-nowrap font-normal text-[14px]'>{typeOptions[type].label}</p>
				{/* <X role='button' className='invisible size-[18px] group-hover:visible rounded-[3px]' onClick={() => changeSource(type, false, applyFn)} /> */}
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
		deleted?: () => void;
		className?: ClassNameValue;
		wrapperClassName?: ClassNameValue;
		deleteButtonClassName?: ClassNameValue;
	}) => {
		return (
			<Tooltip value={text} side='top' delayDuration={1000}>
				<div key={text} className={twMerge('group px-[2px] cursor-default max-w-[320px] bg-white border-[#EEEEEE] border h-[30px] rounded-[5px] flex justify-center items-center w-fit', wrapperClassName)}>
					<div className={twMerge('flex items-center w-full justify-between max-w-full rounded-[3px] h-[calc(100%-4px)] py-[5px] ps-[5px] pe-[6px] gap-[8px]', className)}>
						<div className='flex items-center gap-[8px] leading-[16px]'>
							<img src='icons/text.svg' alt='' />
							<p className='text-nowrap cursor-default max-w-full overflow-hidden text-ellipsis font-light text-[14px]'>{text}</p>
						</div>
						{deleted && (
							<X
								role='button'
								className={twMerge('invisible min-w-[18px] size-[18px] group-hover:visible rounded-[3px]', deleteButtonClassName)}
								onClick={(e) => {
									e.stopPropagation();
									const newContentConditions = contentConditions?.filter((_, i) => i !== index);
									if (apply) {
										setContentConditions(newContentConditions);
										applyFn(sources, level, newContentConditions);
									}
									deleted?.();
								}}
							/>
						)}
					</div>
				</div>
			</Tooltip>
		);
	};

	const DropDownFilter = () => {
		const [dropdownOpen, setDropdownOpen] = useState(false);
		const [sources, setSources] = useState<Type[]>(structuredClone(def?.types || []));
		const [content, setContent] = useState<string[]>(structuredClone(def?.content || []));
		const [level, setLevel] = useState<Level>(def?.level || ALL_LEVELS[ALL_LEVELS.length - 1]);
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

		const changeMenuOpen = () => {
			setDropdownOpen(!dropdownOpen);
			setSources(structuredClone(def?.types || []));
			setContent(structuredClone(def?.content || []));
		};

		return (
			<div className='wrapper relative flex items-center h-[30px]' ref={wrapperRef}>
				<div>
					<div onClick={changeMenuOpen} role='button' className={twMerge('flex group bg-white rounded-[6px] items-center gap-[6px] max-h-[30px] h-[30px] w-[73px] min-w-max pe-[8px]', dropdownOpen && 'bg-white border-transparent')}>
						<img src='icons/funnel.svg' className='[stroke-width:2px] size-[16px]' />
						<p className='text-[14px] group-hover:underline font-medium select-none'>Edit Filters</p>
					</div>
				</div>
				<div className={twMerge('hidden border rounded-[7px] absolute top-[38px] left-0 w-[246px] z-50 bg-white', dropdownOpen && 'block', usePopupToLeft ? 'right-0 left-[unset]' : '')}>
					<div className='flex justify-between items-center'>
						<div className='flex items-center gap-[6px] h-[35px] px-[14px]'>
							{/* <ListFilter className='[stroke-width:2px] size-[16px]' /> */}
							<p className='text-[14px] font-normal'>Filter</p>
						</div>
						<div role='button' onClick={changeMenuOpen} className='flex h-[24px] w-[24px] items-center me-[2px] justify-center'>
							<img src='icons/close.svg' alt='close' />
						</div>
					</div>
					<hr className='bg-[#EBECF0]' />
					<div className='flex gap-[6px] items-center px-[14px]'>
						<p className='text-[14px] font-normal'>Level:</p>
						<Select value={level} onValueChange={(value) => setLevel(value as Level)}>
							<SelectTrigger className='!ring-0 !ring-offset-0 h-[30px] m-auto my-[5px] capitalize border'>
								<SelectValue placeholder={level?.toLowerCase()} />
							</SelectTrigger>
							<SelectContent className='z-[999999]'>
								<SelectGroup>
									{ALL_LEVELS.toReversed().map((level) => (
										<SelectItem key={level} value={level} className='capitalize'>
											{level?.toLowerCase()}
										</SelectItem>
									))}
								</SelectGroup>
							</SelectContent>
						</Select>
					</div>
					<hr className='bg-[#EBECF0]' />
					<div className='flex flex-col gap-[4px] mt-[9px] pb-[11px] px-[8px]'>
						{ALL_TYPES.map((type) => (
							<div key={type} className={twMerge('flex items-center rounded-[3px] h-[24px] py-[4px] ps-[4px] space-x-2 hover:bg-main', sources.includes(type) && '!bg-gray-4')}>
								<Checkbox id={type} checked={sources?.includes(type)} className='[&_svg]:[stroke:#006E53] border-black rounded-[2px] !bg-white' onCheckedChange={(isChecked) => changeSource(type, !!isChecked)} />
								<label className='text-[14px] font-light w-full cursor-pointer flex gap-[8px] !ms-[12px]' htmlFor={type}>
									<img src={typeOptions[type].icon} alt={type} />
									{typeOptions[type].label}
								</label>
							</div>
						))}
					</div>
					<hr className='bg-[#EBECF0]' />
					<div className={twMerge('inputs flex flex-wrap gap-[6px] max-h-[200px] overflow-auto px-[14px] pb-[14px] pt-[11px]', !content?.length && 'h-0 p-0')}>
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
									deleted={() => setContent(content.filter((_, index) => index !== i))}
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
					<div className='buttons flex gap-[8px] items-center h-[47px] p-[6px]'>
						<Button onClick={() => applyFn([], 'DEBUG', [])} variant='ghost' className='flex-1 text-[12px] bg-[#FAFAFA] hover:text-[#151515] hover:bg-[#F3F5F9] font-normal text-[#656565] h-[35px] w-[95px]'>
							Clear all
						</Button>
						<Button
							variant='ghost'
							onClick={() => {
								applyFn(sources, level, content);
								setDropdownOpen(false);
							}}
							className='flex-1 ps-[12px] pe-[10px] text-[12px] font-normal !text-white bg-green-main hover:bg-[#005C3F] w-fit max-w-fit h-[35px]'>
							Apply
						</Button>
					</div>
				</div>
			</div>
		);
	};

	return (
		<div className='flex items-center justify-between pe-[14px] z-[1] bg-white'>
			<div className={twMerge('flex z-[1] pt-[10px] pb-[8px] pe-[12px] ps-[14px] gap-[8px] h-fit min-h-[58px]', (!!def?.types?.length || !!def?.content?.length) && 'min-h-[50px]', className)}>
				<div className='filters-button flex items-start gap-[10px] flex-wrap'>
					{showTags && !!def?.types?.length && def.types.map((type) => <TypeChip key={type} type={type} />)}
					{showTags && def?.content?.map((c: string, index: number) => <CondChip key={c} text={c} index={index} wrapperClassName='cursor-auto' />)}
					{showDropdown && <DropDownFilter />}
				</div>
			</div>
			{deleteFilterTab && (
				<Button onClick={() => deleteFilterTab(filterId)} variant='outline' className='self-start mt-[10px] min-h-[28px] min-w-[28px] size-[28px] p-0 border border-[#EEEEEE] rounded-[6px] shadow-main'>
					<X className='size-[14px] min-h-[14px] min-w-[14px]' />
				</Button>
			)}
		</div>
	);
};

export default memo(LogFilters);
