/* eslint-disable @typescript-eslint/no-explicit-any */
import {twJoin, twMerge} from 'tailwind-merge';
import {Level, Type} from '../log-filters/log-filters';
import {useState} from 'react';
import {Plus} from 'lucide-react';

/**
 * Interface defining the structure of a filter definition
 */
interface DefInterface {
	/** The log level for filtering */
	level?: Level;
	/** Array of log types to filter by */
	types?: Type[];
	/** Array of content strings to filter by */
	content?: string[];
}

/**
 * Interface representing a filter tab with its configuration
 */
export interface Filter {
	/** Unique identifier for the filter */
	id: number;
	/** Display name of the filter */
	name: string;
	/** Filter definition containing the actual filter criteria */
	def: DefInterface | null;
}

/**
 * Props interface for the FilterTabs component
 */
interface FilterTabsFilterProps {
	/** Array of filter tabs to display */
	filterTabs: Filter[];
	/** Function to set the currently active filter tab */
	setCurrFilterTabs: React.Dispatch<React.SetStateAction<number | null>>;
	/** Function to update the filter tabs array */
	setFilterTabs: React.Dispatch<React.SetStateAction<Filter[]>>;
	/** ID of the currently active filter tab */
	currFilterTabs: number | null;
}

/**
 * Component that renders a horizontal list of filter tabs with editing capabilities
 * @param props - The component props
 * @returns JSX element containing the filter tabs interface
 */
const FilterTabs = ({filterTabs, setCurrFilterTabs, setFilterTabs, currFilterTabs}: FilterTabsFilterProps) => {
	const [isEditing, setIsEditing] = useState(false);
	const [inputVal, setInputVal] = useState('');

	/**
	 * Creates a new filter tab and adds it to the list
	 */
	const addFilter = () => {
		const val: Filter = {id: Date.now(), name: 'Logs', def: {level: 'DEBUG', types: []}};
		const allTabs = [...filterTabs, val];
		setFilterTabs(allTabs);
		setCurrFilterTabs(val.id);
	};

	/**
	 * Handles click events on filter tab names to enable editing mode
	 * @param e - The mouse click event
	 * @param tab - The filter tab that was clicked
	 */
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

	/**
	 * Handles completion of filter tab name editing
	 * @param e - The event that triggered the edit completion
	 * @param tab - The filter tab being edited
	 */
	const editFinished = (e: any, tab: Filter) => {
		setIsEditing(false);
		if (!e.target.textContent) e.target.textContent = inputVal || tab.name;
		tab.name = e.target.textContent;
		localStorage.setItem('filters', JSON.stringify(filterTabs));
		e.target.blur();
		const selection = window.getSelection();
		selection?.removeAllRanges();
	};

	/**
	 * Handles cancellation of filter tab name editing
	 * @param e - The event that triggered the edit cancellation
	 * @param tab - The filter tab being edited
	 */
	const editCancelled = (e: any, tab: Filter) => {
		setIsEditing(false);
		e.target.textContent = tab.name;
		e.target.blur();
	};

	return (
		<div className={twMerge('ps-[10px] flex gap-[8px] bg-white items-center min-h-[42px] filter-tabs border-b border-[#EDEFF3] overflow-x-auto z-10 overflow-y-visible no-scrollbar', isEditing && 'border-[#ebecf0]')}>
			{filterTabs.map((tab: Filter) => (
				<div
					className={twJoin(
						'bg-[#FAFAFA] hover:bg-[#F3F5F9] border border-transparent relative rounded-[6px] text-[#A9A9A9] hover:text-[#282828]',
						tab.id === currFilterTabs && 'shadow-main-inset !bg-[#FAFAFA] !text-[#282828]',
						tab.id === currFilterTabs && isEditing && '!border-black !shadow-none'
					)}
					key={tab.id}
					role='button'
					onClick={() => {
						setIsEditing(false);
						setCurrFilterTabs(tab.id);
					}}>
					<div
						className={twJoin(
							'group flex min-h-[28px] max-w-[200px] rounded-[6px] max-h-[28px] justify-center leading-[18px] text-[15px] border border-transparent items-center border-e w-fit',
							tab.id === currFilterTabs && isEditing && 'h-full rounded-[5px]'
						)}>
						<div className={twMerge('flex items-center gap-[8px] relative max-w-full')}>
							<p
								tabIndex={-1}
								onClick={(e) => tab.id === currFilterTabs && clicked(e, tab)}
								contentEditable={tab.id === currFilterTabs && isEditing}
								suppressContentEditableWarning
								onKeyDown={(e) => (e.key === 'Enter' ? editFinished(e, tab) : e.key === 'Escape' && editCancelled(e, tab))}
								onBlur={(e) => editFinished(e, tab)}
								className={twMerge(
									'text-[15px] flex-1 overflow-hidden whitespace-nowrap text-ellipsis h-[28px] px-[8px] outline-none items-center border border-transparent flex !justify-start',
									tab.id === currFilterTabs && !isEditing && 'hover:cursor-text'
								)}>
								{tab.name}
							</p>
							{/* {filterTabs.length > 0 && <img src='icons/close.svg' alt='close' className='h-[20px]' role='button' height={10} width={10} onClick={() => deleteFilterTab(tab.id)} />} */}
						</div>
					</div>
				</div>
			))}
			<div className='flex gap-[10px] items-center justify-center size-[28px] min-w-[28px] w-fit sticky right-0 text-[#151515] hover:text-[#151515] bg-white hover:bg-[#f3f5f9] rounded-[6px]' role='button' onClick={addFilter}>
				<Plus size={16} />
			</div>
		</div>
	);
};

export default FilterTabs;
