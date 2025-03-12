import {useEffect, useRef} from 'react';
import {EditorView, lineNumbers} from '@codemirror/view';
import {EditorState} from '@codemirror/state';

const CodeEditor = ({text}: {text: string}) => {
	const editorRef = useRef<HTMLDivElement | null>(null);

	useEffect(() => {
		if (editorRef.current) {
			const state = EditorState.create({
				doc: text,
				extensions: [lineNumbers(), EditorView.editable.of(false)],
			});

			const view = new EditorView({
				state,
				parent: editorRef.current,
			});

			return () => view.destroy();
		}
	}, [text]);

	return (
		<div
			ref={editorRef}
			contentEditable={false}
			className='[&_.cm-lineNumbers]:bg-white [&_.cm-editor]:outline-none [&_.cm-gutters]:border-0 [&_.cm-gutters]:ps-[0.5em] [&_.cm-gutters]:text-[#bbb] [&_.cm-gutters]:pe-[1.2em] [&_.cm-gutters]:bg-white max-w-full [&_.cm-scroller>div]:max-w-[-webkit-fill-available] [&_.cm-scroller>div:nth-child(2)]:flex-1 [&_.cm-scroller>div]:[white-space:break-spaces]'
		/>
	);
};

export default CodeEditor;
