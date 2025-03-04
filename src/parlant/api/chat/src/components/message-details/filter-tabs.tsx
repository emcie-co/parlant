/* eslint-disable @typescript-eslint/no-explicit-any */
import {twJoin, twMerge} from 'tailwind-merge';
import {Level, Type} from '../log-filters/log-filters';
import {useState} from 'react';
import {Plus, X} from 'lucide-react';

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

interface FilterTabsFilterProps {
	filterTabs: Filter[];
	setCurrFilterTabs: React.Dispatch<React.SetStateAction<number | null>>;
	setFilterTabs: React.Dispatch<React.SetStateAction<Filter[]>>;
	currFilterTabs: number | null;
	setFilters: React.Dispatch<React.SetStateAction<any>>;
}

const FilterTabs = ({filterTabs, setCurrFilterTabs, setFilterTabs, currFilterTabs, setFilters}: FilterTabsFilterProps) => {
	const [isEditing, setIsEditing] = useState(false);
	const [inputVal, setInputVal] = useState('');

	const deleteFilterTab = (id: number, index: number) => {
		const filteredTabs = filterTabs.filter((t) => t.id !== id);
		setFilterTabs(filteredTabs);

		if (currFilterTabs === id && filteredTabs.length) {
			const newTab = filteredTabs?.[(index || 1) - 1]?.id || filteredTabs?.[0]?.id || null;
			setTimeout(() => setCurrFilterTabs(newTab), 0);
		}
		if (!filteredTabs.length) setFilters({});
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
		const selection = window.getSelection();
		selection?.removeAllRanges();
	};

	const editCancelled = (e: any, tab: Filter) => {
		setIsEditing(false);
		e.target.textContent = tab.name;
		e.target.blur();
	};

	return (
		<div className={twMerge('ps-[10px] flex gap-[8px] items-center min-h-[42px] filter-tabs border-b border-[#EDEFF3] overflow-x-auto  overflow-y-visible no-scrollbar', isEditing && 'border-[#ebecf0]')}>
			{filterTabs.map((tab: Filter, i: number) => (
				<div
					className={twJoin(
						'bg-[#FAFAFA] border border-transparent relative rounded-[6px] text-[#A9A9A9]',
						tab.id === currFilterTabs && 'after:content-[""] after:absolute after:left-0 after:w-full after:h-[2px] after:bg-black after:-bottom-[7px] shadow-main-inset text-black',
						tab.id === currFilterTabs && isEditing && '!border-black !shadow-none'
					)}
					key={tab.id}>
					<div
						key={tab.id}
						role='button'
						onClick={() => {
							setIsEditing(false);
							setCurrFilterTabs(tab.id);
						}}
						className={twJoin(
							'group flex min-h-[28px] max-w-[200px] rounded-[6px] max-h-[28px] justify-center leading-[18px] text-[15px] border border-transparent items-center border-e w-fit',
							tab.id === currFilterTabs && isEditing && 'h-full rounded-[5px]'
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
									tab.id === currFilterTabs && !isEditing && 'hover:cursor-text'
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
			<div className='flex gap-[10px] ms-[6px] items-center rounded-[2px] p-[4px] w-fit sticky right-0 text-[#151515] hover:text-[#151515] bg-white' role='button' onClick={addFilter}>
				<Plus size={16} />
			</div>
		</div>
	);
};

export default FilterTabs;
