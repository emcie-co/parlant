import {useEffect, useRef} from 'react';
import {EditorView, lineNumbers} from '@codemirror/view';
import {EditorState} from '@codemirror/state';
import {search, searchKeymap, highlightSelectionMatches, openSearchPanel} from '@codemirror/search';
import {keymap} from '@codemirror/view';

const CodeEditor = ({text}: {text: string}) => {
	const editorRef = useRef<HTMLDivElement | null>(null);
	const editorViewRef = useRef<EditorView | null>(null);

	useEffect(() => {
		if (editorRef.current) {
			const extensions = [lineNumbers(), EditorView.editable.of(false), search({top: true}), highlightSelectionMatches(), keymap.of(searchKeymap)];

			const state = EditorState.create({doc: text, extensions});

			const view = new EditorView({
				state,
				parent: editorRef.current,
			});

			editorViewRef.current = view;

			return () => view.destroy();
		}
	}, [text]);

	useEffect(() => {
		if (editorViewRef.current) setTimeout(() => openSearchPanel(editorViewRef.current as EditorView), 0);
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [editorViewRef.current]);

	return (
		<div
			onKeyDownCapture={(e) => e.key === 'Escape' && e.stopPropagation()}
			className='[&_.cm-search_*]:hidden [&_.cm-gutters]:bg-white [&_.cm-gutters]:text-[#bbb] [&_.cm-gutters]:border-0 [&_.cm-gutters]:ps-[0.5em] [&_.cm-gutters]:pe-[1.2em] [&_.cm-search]:bg-white [&_.cm-scroller>div:nth-child(2)]:flex-1 [&_.cm-scroller>div:nth-child(2)]:[white-space:break-spaces] [&_.cm-panels]:border-none [&_.cm-search_input:first-child]:block [&_.cm-search_input:first-child]:rounded-[5px] [&_.cm-search_input:first-child]:w-[300px] [&_.cm-panels]:sticky [&_.cm-panels]:translate-y-0 [&_.cm-panels]:h-[39px] [&_.cm-panels]:bg-white [&_.cm-panels]:-translate-y-[100%] [&_.cm-panels]:pl-[20px]'>
			<div ref={editorRef} className='your-class-name' />
		</div>
	);
};

export default CodeEditor;
