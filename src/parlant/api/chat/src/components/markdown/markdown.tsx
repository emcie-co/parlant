/* eslint-disable @typescript-eslint/no-unused-vars */
import {Streamdown} from 'streamdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import rehypeRaw from 'rehype-raw';
import 'highlight.js/styles/github.css';
import styles from '../message/message.module.scss';
import {twMerge} from 'tailwind-merge';

function preserveBlankLines(md: string): string {
	return md?.replace?.(/\\n/g, '\n')?.replace(/\n(?!-)/g, '<br/>') || md;
}

// Streamdown is a drop-in for ReactMarkdown that gracefully handles partial
// markdown tokens mid-stream (incomplete code fences, links, emphasis, etc.) —
// so we can render the *currently revealed* prefix of a streaming message or
// status text as live markdown without flicker, instead of waiting for the
// stream to terminate.
const Markdown = ({children, className}: {children: string; className?: string}) => {
	return (
		<Streamdown
			components={{
				p: 'div',
				img: ({node, ...props}) => <img {...props} loading='lazy' alt='' />,
			}}
			rehypePlugins={[rehypeHighlight, rehypeRaw]}
			remarkPlugins={[remarkGfm]}
			parseIncompleteMarkdown
			className={twMerge('leading-[19px]', styles.markdown, className)}>
			{preserveBlankLines(children)}
		</Streamdown>
	);
};

export default Markdown;
