const __vite__mapDeps = (i, m = __vite__mapDeps, d = m.f || (m.f = ['assets/chat-BI4eCUxH.js', 'assets/chat-CrQIcuHW.css'])) => i.map((i) => d[i]);
function Iv(e, t) {
	for (var n = 0; n < t.length; n++) {
		const r = t[n];
		if (typeof r != 'string' && !Array.isArray(r)) {
			for (const o in r)
				if (o !== 'default' && !(o in e)) {
					const i = Object.getOwnPropertyDescriptor(r, o);
					i && Object.defineProperty(e, o, i.get ? i : {enumerable: !0, get: () => r[o]});
				}
		}
	}
	return Object.freeze(Object.defineProperty(e, Symbol.toStringTag, {value: 'Module'}));
}
(function () {
	const t = document.createElement('link').relList;
	if (t && t.supports && t.supports('modulepreload')) return;
	for (const o of document.querySelectorAll('link[rel="modulepreload"]')) r(o);
	new MutationObserver((o) => {
		for (const i of o) if (i.type === 'childList') for (const s of i.addedNodes) s.tagName === 'LINK' && s.rel === 'modulepreload' && r(s);
	}).observe(document, {childList: !0, subtree: !0});
	function n(o) {
		const i = {};
		return (
			o.integrity && (i.integrity = o.integrity),
			o.referrerPolicy && (i.referrerPolicy = o.referrerPolicy),
			o.crossOrigin === 'use-credentials' ? (i.credentials = 'include') : o.crossOrigin === 'anonymous' ? (i.credentials = 'omit') : (i.credentials = 'same-origin'),
			i
		);
	}
	function r(o) {
		if (o.ep) return;
		o.ep = !0;
		const i = n(o);
		fetch(o.href, i);
	}
})();
var xk = typeof globalThis < 'u' ? globalThis : typeof window < 'u' ? window : typeof global < 'u' ? global : typeof self < 'u' ? self : {};
function Ar(e) {
	return e && e.__esModule && Object.prototype.hasOwnProperty.call(e, 'default') ? e.default : e;
}
var Ef = {exports: {}},
	hs = {},
	Cf = {exports: {}},
	G = {};
/**
 * @license React
 * react.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */ var Lo = Symbol.for('react.element'),
	Dv = Symbol.for('react.portal'),
	Lv = Symbol.for('react.fragment'),
	Fv = Symbol.for('react.strict_mode'),
	zv = Symbol.for('react.profiler'),
	$v = Symbol.for('react.provider'),
	Bv = Symbol.for('react.context'),
	Uv = Symbol.for('react.forward_ref'),
	Hv = Symbol.for('react.suspense'),
	Wv = Symbol.for('react.memo'),
	Vv = Symbol.for('react.lazy'),
	_c = Symbol.iterator;
function Gv(e) {
	return e === null || typeof e != 'object' ? null : ((e = (_c && e[_c]) || e['@@iterator']), typeof e == 'function' ? e : null);
}
var bf = {
		isMounted: function () {
			return !1;
		},
		enqueueForceUpdate: function () {},
		enqueueReplaceState: function () {},
		enqueueSetState: function () {},
	},
	kf = Object.assign,
	Tf = {};
function Mr(e, t, n) {
	(this.props = e), (this.context = t), (this.refs = Tf), (this.updater = n || bf);
}
Mr.prototype.isReactComponent = {};
Mr.prototype.setState = function (e, t) {
	if (typeof e != 'object' && typeof e != 'function' && e != null) throw Error('setState(...): takes an object of state variables to update or a function which returns an object of state variables.');
	this.updater.enqueueSetState(this, e, t, 'setState');
};
Mr.prototype.forceUpdate = function (e) {
	this.updater.enqueueForceUpdate(this, e, 'forceUpdate');
};
function Pf() {}
Pf.prototype = Mr.prototype;
function Ja(e, t, n) {
	(this.props = e), (this.context = t), (this.refs = Tf), (this.updater = n || bf);
}
var eu = (Ja.prototype = new Pf());
eu.constructor = Ja;
kf(eu, Mr.prototype);
eu.isPureReactComponent = !0;
var Rc = Array.isArray,
	Nf = Object.prototype.hasOwnProperty,
	tu = {current: null},
	_f = {key: !0, ref: !0, __self: !0, __source: !0};
function Rf(e, t, n) {
	var r,
		o = {},
		i = null,
		s = null;
	if (t != null) for (r in (t.ref !== void 0 && (s = t.ref), t.key !== void 0 && (i = '' + t.key), t)) Nf.call(t, r) && !_f.hasOwnProperty(r) && (o[r] = t[r]);
	var l = arguments.length - 2;
	if (l === 1) o.children = n;
	else if (1 < l) {
		for (var a = Array(l), u = 0; u < l; u++) a[u] = arguments[u + 2];
		o.children = a;
	}
	if (e && e.defaultProps) for (r in ((l = e.defaultProps), l)) o[r] === void 0 && (o[r] = l[r]);
	return {$$typeof: Lo, type: e, key: i, ref: s, props: o, _owner: tu.current};
}
function Kv(e, t) {
	return {$$typeof: Lo, type: e.type, key: t, ref: e.ref, props: e.props, _owner: e._owner};
}
function nu(e) {
	return typeof e == 'object' && e !== null && e.$$typeof === Lo;
}
function Yv(e) {
	var t = {'=': '=0', ':': '=2'};
	return (
		'$' +
		e.replace(/[=:]/g, function (n) {
			return t[n];
		})
	);
}
var Ac = /\/+/g;
function nl(e, t) {
	return typeof e == 'object' && e !== null && e.key != null ? Yv('' + e.key) : t.toString(36);
}
function xi(e, t, n, r, o) {
	var i = typeof e;
	(i === 'undefined' || i === 'boolean') && (e = null);
	var s = !1;
	if (e === null) s = !0;
	else
		switch (i) {
			case 'string':
			case 'number':
				s = !0;
				break;
			case 'object':
				switch (e.$$typeof) {
					case Lo:
					case Dv:
						s = !0;
				}
		}
	if (s)
		return (
			(s = e),
			(o = o(s)),
			(e = r === '' ? '.' + nl(s, 0) : r),
			Rc(o)
				? ((n = ''),
				  e != null && (n = e.replace(Ac, '$&/') + '/'),
				  xi(o, t, n, '', function (u) {
						return u;
				  }))
				: o != null && (nu(o) && (o = Kv(o, n + (!o.key || (s && s.key === o.key) ? '' : ('' + o.key).replace(Ac, '$&/') + '/') + e)), t.push(o)),
			1
		);
	if (((s = 0), (r = r === '' ? '.' : r + ':'), Rc(e)))
		for (var l = 0; l < e.length; l++) {
			i = e[l];
			var a = r + nl(i, l);
			s += xi(i, t, n, a, o);
		}
	else if (((a = Gv(e)), typeof a == 'function')) for (e = a.call(e), l = 0; !(i = e.next()).done; ) (i = i.value), (a = r + nl(i, l++)), (s += xi(i, t, n, a, o));
	else if (i === 'object')
		throw (
			((t = String(e)), Error('Objects are not valid as a React child (found: ' + (t === '[object Object]' ? 'object with keys {' + Object.keys(e).join(', ') + '}' : t) + '). If you meant to render a collection of children, use an array instead.'))
		);
	return s;
}
function Xo(e, t, n) {
	if (e == null) return e;
	var r = [],
		o = 0;
	return (
		xi(e, r, '', '', function (i) {
			return t.call(n, i, o++);
		}),
		r
	);
}
function Qv(e) {
	if (e._status === -1) {
		var t = e._result;
		(t = t()),
			t.then(
				function (n) {
					(e._status === 0 || e._status === -1) && ((e._status = 1), (e._result = n));
				},
				function (n) {
					(e._status === 0 || e._status === -1) && ((e._status = 2), (e._result = n));
				}
			),
			e._status === -1 && ((e._status = 0), (e._result = t));
	}
	if (e._status === 1) return e._result.default;
	throw e._result;
}
var Ie = {current: null},
	Si = {transition: null},
	Xv = {ReactCurrentDispatcher: Ie, ReactCurrentBatchConfig: Si, ReactCurrentOwner: tu};
function Af() {
	throw Error('act(...) is not supported in production builds of React.');
}
G.Children = {
	map: Xo,
	forEach: function (e, t, n) {
		Xo(
			e,
			function () {
				t.apply(this, arguments);
			},
			n
		);
	},
	count: function (e) {
		var t = 0;
		return (
			Xo(e, function () {
				t++;
			}),
			t
		);
	},
	toArray: function (e) {
		return (
			Xo(e, function (t) {
				return t;
			}) || []
		);
	},
	only: function (e) {
		if (!nu(e)) throw Error('React.Children.only expected to receive a single React element child.');
		return e;
	},
};
G.Component = Mr;
G.Fragment = Lv;
G.Profiler = zv;
G.PureComponent = Ja;
G.StrictMode = Fv;
G.Suspense = Hv;
G.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = Xv;
G.act = Af;
G.cloneElement = function (e, t, n) {
	if (e == null) throw Error('React.cloneElement(...): The argument must be a React element, but you passed ' + e + '.');
	var r = kf({}, e.props),
		o = e.key,
		i = e.ref,
		s = e._owner;
	if (t != null) {
		if ((t.ref !== void 0 && ((i = t.ref), (s = tu.current)), t.key !== void 0 && (o = '' + t.key), e.type && e.type.defaultProps)) var l = e.type.defaultProps;
		for (a in t) Nf.call(t, a) && !_f.hasOwnProperty(a) && (r[a] = t[a] === void 0 && l !== void 0 ? l[a] : t[a]);
	}
	var a = arguments.length - 2;
	if (a === 1) r.children = n;
	else if (1 < a) {
		l = Array(a);
		for (var u = 0; u < a; u++) l[u] = arguments[u + 2];
		r.children = l;
	}
	return {$$typeof: Lo, type: e.type, key: o, ref: i, props: r, _owner: s};
};
G.createContext = function (e) {
	return (e = {$$typeof: Bv, _currentValue: e, _currentValue2: e, _threadCount: 0, Provider: null, Consumer: null, _defaultValue: null, _globalName: null}), (e.Provider = {$$typeof: $v, _context: e}), (e.Consumer = e);
};
G.createElement = Rf;
G.createFactory = function (e) {
	var t = Rf.bind(null, e);
	return (t.type = e), t;
};
G.createRef = function () {
	return {current: null};
};
G.forwardRef = function (e) {
	return {$$typeof: Uv, render: e};
};
G.isValidElement = nu;
G.lazy = function (e) {
	return {$$typeof: Vv, _payload: {_status: -1, _result: e}, _init: Qv};
};
G.memo = function (e, t) {
	return {$$typeof: Wv, type: e, compare: t === void 0 ? null : t};
};
G.startTransition = function (e) {
	var t = Si.transition;
	Si.transition = {};
	try {
		e();
	} finally {
		Si.transition = t;
	}
};
G.unstable_act = Af;
G.useCallback = function (e, t) {
	return Ie.current.useCallback(e, t);
};
G.useContext = function (e) {
	return Ie.current.useContext(e);
};
G.useDebugValue = function () {};
G.useDeferredValue = function (e) {
	return Ie.current.useDeferredValue(e);
};
G.useEffect = function (e, t) {
	return Ie.current.useEffect(e, t);
};
G.useId = function () {
	return Ie.current.useId();
};
G.useImperativeHandle = function (e, t, n) {
	return Ie.current.useImperativeHandle(e, t, n);
};
G.useInsertionEffect = function (e, t) {
	return Ie.current.useInsertionEffect(e, t);
};
G.useLayoutEffect = function (e, t) {
	return Ie.current.useLayoutEffect(e, t);
};
G.useMemo = function (e, t) {
	return Ie.current.useMemo(e, t);
};
G.useReducer = function (e, t, n) {
	return Ie.current.useReducer(e, t, n);
};
G.useRef = function (e) {
	return Ie.current.useRef(e);
};
G.useState = function (e) {
	return Ie.current.useState(e);
};
G.useSyncExternalStore = function (e, t, n) {
	return Ie.current.useSyncExternalStore(e, t, n);
};
G.useTransition = function () {
	return Ie.current.useTransition();
};
G.version = '18.3.1';
Cf.exports = G;
var p = Cf.exports;
const A = Ar(p),
	Zv = Iv({__proto__: null, default: A}, [p]);
/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */ var qv = p,
	Jv = Symbol.for('react.element'),
	ey = Symbol.for('react.fragment'),
	ty = Object.prototype.hasOwnProperty,
	ny = qv.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,
	ry = {key: !0, ref: !0, __self: !0, __source: !0};
function Mf(e, t, n) {
	var r,
		o = {},
		i = null,
		s = null;
	n !== void 0 && (i = '' + n), t.key !== void 0 && (i = '' + t.key), t.ref !== void 0 && (s = t.ref);
	for (r in t) ty.call(t, r) && !ry.hasOwnProperty(r) && (o[r] = t[r]);
	if (e && e.defaultProps) for (r in ((t = e.defaultProps), t)) o[r] === void 0 && (o[r] = t[r]);
	return {$$typeof: Jv, type: e, key: i, ref: s, props: o, _owner: ny.current};
}
hs.Fragment = ey;
hs.jsx = Mf;
hs.jsxs = Mf;
Ef.exports = hs;
var v = Ef.exports,
	Of = {exports: {}},
	Ze = {},
	jf = {exports: {}},
	If = {};
/**
 * @license React
 * scheduler.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */ (function (e) {
	function t(b, P) {
		var j = b.length;
		b.push(P);
		e: for (; 0 < j; ) {
			var O = (j - 1) >>> 1,
				L = b[O];
			if (0 < o(L, P)) (b[O] = P), (b[j] = L), (j = O);
			else break e;
		}
	}
	function n(b) {
		return b.length === 0 ? null : b[0];
	}
	function r(b) {
		if (b.length === 0) return null;
		var P = b[0],
			j = b.pop();
		if (j !== P) {
			b[0] = j;
			e: for (var O = 0, L = b.length, Y = L >>> 1; O < Y; ) {
				var ie = 2 * (O + 1) - 1,
					Ne = b[ie],
					K = ie + 1,
					xe = b[K];
				if (0 > o(Ne, j)) K < L && 0 > o(xe, Ne) ? ((b[O] = xe), (b[K] = j), (O = K)) : ((b[O] = Ne), (b[ie] = j), (O = ie));
				else if (K < L && 0 > o(xe, j)) (b[O] = xe), (b[K] = j), (O = K);
				else break e;
			}
		}
		return P;
	}
	function o(b, P) {
		var j = b.sortIndex - P.sortIndex;
		return j !== 0 ? j : b.id - P.id;
	}
	if (typeof performance == 'object' && typeof performance.now == 'function') {
		var i = performance;
		e.unstable_now = function () {
			return i.now();
		};
	} else {
		var s = Date,
			l = s.now();
		e.unstable_now = function () {
			return s.now() - l;
		};
	}
	var a = [],
		u = [],
		f = 1,
		d = null,
		c = 3,
		g = !1,
		w = !1,
		y = !1,
		S = typeof setTimeout == 'function' ? setTimeout : null,
		m = typeof clearTimeout == 'function' ? clearTimeout : null,
		h = typeof setImmediate < 'u' ? setImmediate : null;
	typeof navigator < 'u' && navigator.scheduling !== void 0 && navigator.scheduling.isInputPending !== void 0 && navigator.scheduling.isInputPending.bind(navigator.scheduling);
	function x(b) {
		for (var P = n(u); P !== null; ) {
			if (P.callback === null) r(u);
			else if (P.startTime <= b) r(u), (P.sortIndex = P.expirationTime), t(a, P);
			else break;
			P = n(u);
		}
	}
	function E(b) {
		if (((y = !1), x(b), !w))
			if (n(a) !== null) (w = !0), F(C);
			else {
				var P = n(u);
				P !== null && X(E, P.startTime - b);
			}
	}
	function C(b, P) {
		(w = !1), y && ((y = !1), m(N), (N = -1)), (g = !0);
		var j = c;
		try {
			for (x(P), d = n(a); d !== null && (!(d.expirationTime > P) || (b && !z())); ) {
				var O = d.callback;
				if (typeof O == 'function') {
					(d.callback = null), (c = d.priorityLevel);
					var L = O(d.expirationTime <= P);
					(P = e.unstable_now()), typeof L == 'function' ? (d.callback = L) : d === n(a) && r(a), x(P);
				} else r(a);
				d = n(a);
			}
			if (d !== null) var Y = !0;
			else {
				var ie = n(u);
				ie !== null && X(E, ie.startTime - P), (Y = !1);
			}
			return Y;
		} finally {
			(d = null), (c = j), (g = !1);
		}
	}
	var T = !1,
		k = null,
		N = -1,
		D = 5,
		M = -1;
	function z() {
		return !(e.unstable_now() - M < D);
	}
	function R() {
		if (k !== null) {
			var b = e.unstable_now();
			M = b;
			var P = !0;
			try {
				P = k(!0, b);
			} finally {
				P ? $() : ((T = !1), (k = null));
			}
		} else T = !1;
	}
	var $;
	if (typeof h == 'function')
		$ = function () {
			h(R);
		};
	else if (typeof MessageChannel < 'u') {
		var U = new MessageChannel(),
			Q = U.port2;
		(U.port1.onmessage = R),
			($ = function () {
				Q.postMessage(null);
			});
	} else
		$ = function () {
			S(R, 0);
		};
	function F(b) {
		(k = b), T || ((T = !0), $());
	}
	function X(b, P) {
		N = S(function () {
			b(e.unstable_now());
		}, P);
	}
	(e.unstable_IdlePriority = 5),
		(e.unstable_ImmediatePriority = 1),
		(e.unstable_LowPriority = 4),
		(e.unstable_NormalPriority = 3),
		(e.unstable_Profiling = null),
		(e.unstable_UserBlockingPriority = 2),
		(e.unstable_cancelCallback = function (b) {
			b.callback = null;
		}),
		(e.unstable_continueExecution = function () {
			w || g || ((w = !0), F(C));
		}),
		(e.unstable_forceFrameRate = function (b) {
			0 > b || 125 < b ? console.error('forceFrameRate takes a positive int between 0 and 125, forcing frame rates higher than 125 fps is not supported') : (D = 0 < b ? Math.floor(1e3 / b) : 5);
		}),
		(e.unstable_getCurrentPriorityLevel = function () {
			return c;
		}),
		(e.unstable_getFirstCallbackNode = function () {
			return n(a);
		}),
		(e.unstable_next = function (b) {
			switch (c) {
				case 1:
				case 2:
				case 3:
					var P = 3;
					break;
				default:
					P = c;
			}
			var j = c;
			c = P;
			try {
				return b();
			} finally {
				c = j;
			}
		}),
		(e.unstable_pauseExecution = function () {}),
		(e.unstable_requestPaint = function () {}),
		(e.unstable_runWithPriority = function (b, P) {
			switch (b) {
				case 1:
				case 2:
				case 3:
				case 4:
				case 5:
					break;
				default:
					b = 3;
			}
			var j = c;
			c = b;
			try {
				return P();
			} finally {
				c = j;
			}
		}),
		(e.unstable_scheduleCallback = function (b, P, j) {
			var O = e.unstable_now();
			switch ((typeof j == 'object' && j !== null ? ((j = j.delay), (j = typeof j == 'number' && 0 < j ? O + j : O)) : (j = O), b)) {
				case 1:
					var L = -1;
					break;
				case 2:
					L = 250;
					break;
				case 5:
					L = 1073741823;
					break;
				case 4:
					L = 1e4;
					break;
				default:
					L = 5e3;
			}
			return (
				(L = j + L),
				(b = {id: f++, callback: P, priorityLevel: b, startTime: j, expirationTime: L, sortIndex: -1}),
				j > O ? ((b.sortIndex = j), t(u, b), n(a) === null && b === n(u) && (y ? (m(N), (N = -1)) : (y = !0), X(E, j - O))) : ((b.sortIndex = L), t(a, b), w || g || ((w = !0), F(C))),
				b
			);
		}),
		(e.unstable_shouldYield = z),
		(e.unstable_wrapCallback = function (b) {
			var P = c;
			return function () {
				var j = c;
				c = P;
				try {
					return b.apply(this, arguments);
				} finally {
					c = j;
				}
			};
		});
})(If);
jf.exports = If;
var oy = jf.exports;
/**
 * @license React
 * react-dom.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */ var iy = p,
	Xe = oy;
function _(e) {
	for (var t = 'https://reactjs.org/docs/error-decoder.html?invariant=' + e, n = 1; n < arguments.length; n++) t += '&args[]=' + encodeURIComponent(arguments[n]);
	return 'Minified React error #' + e + '; visit ' + t + ' for the full message or use the non-minified dev environment for full errors and additional helpful warnings.';
}
var Df = new Set(),
	po = {};
function $n(e, t) {
	Sr(e, t), Sr(e + 'Capture', t);
}
function Sr(e, t) {
	for (po[e] = t, e = 0; e < t.length; e++) Df.add(t[e]);
}
var Lt = !(typeof window > 'u' || typeof window.document > 'u' || typeof window.document.createElement > 'u'),
	Ul = Object.prototype.hasOwnProperty,
	sy =
		/^[:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD][:A-Z_a-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\-.0-9\u00B7\u0300-\u036F\u203F-\u2040]*$/,
	Mc = {},
	Oc = {};
function ly(e) {
	return Ul.call(Oc, e) ? !0 : Ul.call(Mc, e) ? !1 : sy.test(e) ? (Oc[e] = !0) : ((Mc[e] = !0), !1);
}
function ay(e, t, n, r) {
	if (n !== null && n.type === 0) return !1;
	switch (typeof t) {
		case 'function':
		case 'symbol':
			return !0;
		case 'boolean':
			return r ? !1 : n !== null ? !n.acceptsBooleans : ((e = e.toLowerCase().slice(0, 5)), e !== 'data-' && e !== 'aria-');
		default:
			return !1;
	}
}
function uy(e, t, n, r) {
	if (t === null || typeof t > 'u' || ay(e, t, n, r)) return !0;
	if (r) return !1;
	if (n !== null)
		switch (n.type) {
			case 3:
				return !t;
			case 4:
				return t === !1;
			case 5:
				return isNaN(t);
			case 6:
				return isNaN(t) || 1 > t;
		}
	return !1;
}
function De(e, t, n, r, o, i, s) {
	(this.acceptsBooleans = t === 2 || t === 3 || t === 4), (this.attributeName = r), (this.attributeNamespace = o), (this.mustUseProperty = n), (this.propertyName = e), (this.type = t), (this.sanitizeURL = i), (this.removeEmptyString = s);
}
var Pe = {};
'children dangerouslySetInnerHTML defaultValue defaultChecked innerHTML suppressContentEditableWarning suppressHydrationWarning style'.split(' ').forEach(function (e) {
	Pe[e] = new De(e, 0, !1, e, null, !1, !1);
});
[
	['acceptCharset', 'accept-charset'],
	['className', 'class'],
	['htmlFor', 'for'],
	['httpEquiv', 'http-equiv'],
].forEach(function (e) {
	var t = e[0];
	Pe[t] = new De(t, 1, !1, e[1], null, !1, !1);
});
['contentEditable', 'draggable', 'spellCheck', 'value'].forEach(function (e) {
	Pe[e] = new De(e, 2, !1, e.toLowerCase(), null, !1, !1);
});
['autoReverse', 'externalResourcesRequired', 'focusable', 'preserveAlpha'].forEach(function (e) {
	Pe[e] = new De(e, 2, !1, e, null, !1, !1);
});
'allowFullScreen async autoFocus autoPlay controls default defer disabled disablePictureInPicture disableRemotePlayback formNoValidate hidden loop noModule noValidate open playsInline readOnly required reversed scoped seamless itemScope'
	.split(' ')
	.forEach(function (e) {
		Pe[e] = new De(e, 3, !1, e.toLowerCase(), null, !1, !1);
	});
['checked', 'multiple', 'muted', 'selected'].forEach(function (e) {
	Pe[e] = new De(e, 3, !0, e, null, !1, !1);
});
['capture', 'download'].forEach(function (e) {
	Pe[e] = new De(e, 4, !1, e, null, !1, !1);
});
['cols', 'rows', 'size', 'span'].forEach(function (e) {
	Pe[e] = new De(e, 6, !1, e, null, !1, !1);
});
['rowSpan', 'start'].forEach(function (e) {
	Pe[e] = new De(e, 5, !1, e.toLowerCase(), null, !1, !1);
});
var ru = /[\-:]([a-z])/g;
function ou(e) {
	return e[1].toUpperCase();
}
'accent-height alignment-baseline arabic-form baseline-shift cap-height clip-path clip-rule color-interpolation color-interpolation-filters color-profile color-rendering dominant-baseline enable-background fill-opacity fill-rule flood-color flood-opacity font-family font-size font-size-adjust font-stretch font-style font-variant font-weight glyph-name glyph-orientation-horizontal glyph-orientation-vertical horiz-adv-x horiz-origin-x image-rendering letter-spacing lighting-color marker-end marker-mid marker-start overline-position overline-thickness paint-order panose-1 pointer-events rendering-intent shape-rendering stop-color stop-opacity strikethrough-position strikethrough-thickness stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin stroke-miterlimit stroke-opacity stroke-width text-anchor text-decoration text-rendering underline-position underline-thickness unicode-bidi unicode-range units-per-em v-alphabetic v-hanging v-ideographic v-mathematical vector-effect vert-adv-y vert-origin-x vert-origin-y word-spacing writing-mode xmlns:xlink x-height'
	.split(' ')
	.forEach(function (e) {
		var t = e.replace(ru, ou);
		Pe[t] = new De(t, 1, !1, e, null, !1, !1);
	});
'xlink:actuate xlink:arcrole xlink:role xlink:show xlink:title xlink:type'.split(' ').forEach(function (e) {
	var t = e.replace(ru, ou);
	Pe[t] = new De(t, 1, !1, e, 'http://www.w3.org/1999/xlink', !1, !1);
});
['xml:base', 'xml:lang', 'xml:space'].forEach(function (e) {
	var t = e.replace(ru, ou);
	Pe[t] = new De(t, 1, !1, e, 'http://www.w3.org/XML/1998/namespace', !1, !1);
});
['tabIndex', 'crossOrigin'].forEach(function (e) {
	Pe[e] = new De(e, 1, !1, e.toLowerCase(), null, !1, !1);
});
Pe.xlinkHref = new De('xlinkHref', 1, !1, 'xlink:href', 'http://www.w3.org/1999/xlink', !0, !1);
['src', 'href', 'action', 'formAction'].forEach(function (e) {
	Pe[e] = new De(e, 1, !1, e.toLowerCase(), null, !0, !0);
});
function iu(e, t, n, r) {
	var o = Pe.hasOwnProperty(t) ? Pe[t] : null;
	(o !== null ? o.type !== 0 : r || !(2 < t.length) || (t[0] !== 'o' && t[0] !== 'O') || (t[1] !== 'n' && t[1] !== 'N')) &&
		(uy(t, n, o, r) && (n = null),
		r || o === null
			? ly(t) && (n === null ? e.removeAttribute(t) : e.setAttribute(t, '' + n))
			: o.mustUseProperty
			? (e[o.propertyName] = n === null ? (o.type === 3 ? !1 : '') : n)
			: ((t = o.attributeName), (r = o.attributeNamespace), n === null ? e.removeAttribute(t) : ((o = o.type), (n = o === 3 || (o === 4 && n === !0) ? '' : '' + n), r ? e.setAttributeNS(r, t, n) : e.setAttribute(t, n))));
}
var Ht = iy.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED,
	Zo = Symbol.for('react.element'),
	Jn = Symbol.for('react.portal'),
	er = Symbol.for('react.fragment'),
	su = Symbol.for('react.strict_mode'),
	Hl = Symbol.for('react.profiler'),
	Lf = Symbol.for('react.provider'),
	Ff = Symbol.for('react.context'),
	lu = Symbol.for('react.forward_ref'),
	Wl = Symbol.for('react.suspense'),
	Vl = Symbol.for('react.suspense_list'),
	au = Symbol.for('react.memo'),
	Zt = Symbol.for('react.lazy'),
	zf = Symbol.for('react.offscreen'),
	jc = Symbol.iterator;
function zr(e) {
	return e === null || typeof e != 'object' ? null : ((e = (jc && e[jc]) || e['@@iterator']), typeof e == 'function' ? e : null);
}
var de = Object.assign,
	rl;
function qr(e) {
	if (rl === void 0)
		try {
			throw Error();
		} catch (n) {
			var t = n.stack.trim().match(/\n( *(at )?)/);
			rl = (t && t[1]) || '';
		}
	return (
		`
` +
		rl +
		e
	);
}
var ol = !1;
function il(e, t) {
	if (!e || ol) return '';
	ol = !0;
	var n = Error.prepareStackTrace;
	Error.prepareStackTrace = void 0;
	try {
		if (t)
			if (
				((t = function () {
					throw Error();
				}),
				Object.defineProperty(t.prototype, 'props', {
					set: function () {
						throw Error();
					},
				}),
				typeof Reflect == 'object' && Reflect.construct)
			) {
				try {
					Reflect.construct(t, []);
				} catch (u) {
					var r = u;
				}
				Reflect.construct(e, [], t);
			} else {
				try {
					t.call();
				} catch (u) {
					r = u;
				}
				e.call(t.prototype);
			}
		else {
			try {
				throw Error();
			} catch (u) {
				r = u;
			}
			e();
		}
	} catch (u) {
		if (u && r && typeof u.stack == 'string') {
			for (
				var o = u.stack.split(`
`),
					i = r.stack.split(`
`),
					s = o.length - 1,
					l = i.length - 1;
				1 <= s && 0 <= l && o[s] !== i[l];

			)
				l--;
			for (; 1 <= s && 0 <= l; s--, l--)
				if (o[s] !== i[l]) {
					if (s !== 1 || l !== 1)
						do
							if ((s--, l--, 0 > l || o[s] !== i[l])) {
								var a =
									`
` + o[s].replace(' at new ', ' at ');
								return e.displayName && a.includes('<anonymous>') && (a = a.replace('<anonymous>', e.displayName)), a;
							}
						while (1 <= s && 0 <= l);
					break;
				}
		}
	} finally {
		(ol = !1), (Error.prepareStackTrace = n);
	}
	return (e = e ? e.displayName || e.name : '') ? qr(e) : '';
}
function cy(e) {
	switch (e.tag) {
		case 5:
			return qr(e.type);
		case 16:
			return qr('Lazy');
		case 13:
			return qr('Suspense');
		case 19:
			return qr('SuspenseList');
		case 0:
		case 2:
		case 15:
			return (e = il(e.type, !1)), e;
		case 11:
			return (e = il(e.type.render, !1)), e;
		case 1:
			return (e = il(e.type, !0)), e;
		default:
			return '';
	}
}
function Gl(e) {
	if (e == null) return null;
	if (typeof e == 'function') return e.displayName || e.name || null;
	if (typeof e == 'string') return e;
	switch (e) {
		case er:
			return 'Fragment';
		case Jn:
			return 'Portal';
		case Hl:
			return 'Profiler';
		case su:
			return 'StrictMode';
		case Wl:
			return 'Suspense';
		case Vl:
			return 'SuspenseList';
	}
	if (typeof e == 'object')
		switch (e.$$typeof) {
			case Ff:
				return (e.displayName || 'Context') + '.Consumer';
			case Lf:
				return (e._context.displayName || 'Context') + '.Provider';
			case lu:
				var t = e.render;
				return (e = e.displayName), e || ((e = t.displayName || t.name || ''), (e = e !== '' ? 'ForwardRef(' + e + ')' : 'ForwardRef')), e;
			case au:
				return (t = e.displayName || null), t !== null ? t : Gl(e.type) || 'Memo';
			case Zt:
				(t = e._payload), (e = e._init);
				try {
					return Gl(e(t));
				} catch {}
		}
	return null;
}
function dy(e) {
	var t = e.type;
	switch (e.tag) {
		case 24:
			return 'Cache';
		case 9:
			return (t.displayName || 'Context') + '.Consumer';
		case 10:
			return (t._context.displayName || 'Context') + '.Provider';
		case 18:
			return 'DehydratedFragment';
		case 11:
			return (e = t.render), (e = e.displayName || e.name || ''), t.displayName || (e !== '' ? 'ForwardRef(' + e + ')' : 'ForwardRef');
		case 7:
			return 'Fragment';
		case 5:
			return t;
		case 4:
			return 'Portal';
		case 3:
			return 'Root';
		case 6:
			return 'Text';
		case 16:
			return Gl(t);
		case 8:
			return t === su ? 'StrictMode' : 'Mode';
		case 22:
			return 'Offscreen';
		case 12:
			return 'Profiler';
		case 21:
			return 'Scope';
		case 13:
			return 'Suspense';
		case 19:
			return 'SuspenseList';
		case 25:
			return 'TracingMarker';
		case 1:
		case 0:
		case 17:
		case 2:
		case 14:
		case 15:
			if (typeof t == 'function') return t.displayName || t.name || null;
			if (typeof t == 'string') return t;
	}
	return null;
}
function fn(e) {
	switch (typeof e) {
		case 'boolean':
		case 'number':
		case 'string':
		case 'undefined':
			return e;
		case 'object':
			return e;
		default:
			return '';
	}
}
function $f(e) {
	var t = e.type;
	return (e = e.nodeName) && e.toLowerCase() === 'input' && (t === 'checkbox' || t === 'radio');
}
function fy(e) {
	var t = $f(e) ? 'checked' : 'value',
		n = Object.getOwnPropertyDescriptor(e.constructor.prototype, t),
		r = '' + e[t];
	if (!e.hasOwnProperty(t) && typeof n < 'u' && typeof n.get == 'function' && typeof n.set == 'function') {
		var o = n.get,
			i = n.set;
		return (
			Object.defineProperty(e, t, {
				configurable: !0,
				get: function () {
					return o.call(this);
				},
				set: function (s) {
					(r = '' + s), i.call(this, s);
				},
			}),
			Object.defineProperty(e, t, {enumerable: n.enumerable}),
			{
				getValue: function () {
					return r;
				},
				setValue: function (s) {
					r = '' + s;
				},
				stopTracking: function () {
					(e._valueTracker = null), delete e[t];
				},
			}
		);
	}
}
function qo(e) {
	e._valueTracker || (e._valueTracker = fy(e));
}
function Bf(e) {
	if (!e) return !1;
	var t = e._valueTracker;
	if (!t) return !0;
	var n = t.getValue(),
		r = '';
	return e && (r = $f(e) ? (e.checked ? 'true' : 'false') : e.value), (e = r), e !== n ? (t.setValue(e), !0) : !1;
}
function Di(e) {
	if (((e = e || (typeof document < 'u' ? document : void 0)), typeof e > 'u')) return null;
	try {
		return e.activeElement || e.body;
	} catch {
		return e.body;
	}
}
function Kl(e, t) {
	var n = t.checked;
	return de({}, t, {defaultChecked: void 0, defaultValue: void 0, value: void 0, checked: n ?? e._wrapperState.initialChecked});
}
function Ic(e, t) {
	var n = t.defaultValue == null ? '' : t.defaultValue,
		r = t.checked != null ? t.checked : t.defaultChecked;
	(n = fn(t.value != null ? t.value : n)), (e._wrapperState = {initialChecked: r, initialValue: n, controlled: t.type === 'checkbox' || t.type === 'radio' ? t.checked != null : t.value != null});
}
function Uf(e, t) {
	(t = t.checked), t != null && iu(e, 'checked', t, !1);
}
function Yl(e, t) {
	Uf(e, t);
	var n = fn(t.value),
		r = t.type;
	if (n != null) r === 'number' ? ((n === 0 && e.value === '') || e.value != n) && (e.value = '' + n) : e.value !== '' + n && (e.value = '' + n);
	else if (r === 'submit' || r === 'reset') {
		e.removeAttribute('value');
		return;
	}
	t.hasOwnProperty('value') ? Ql(e, t.type, n) : t.hasOwnProperty('defaultValue') && Ql(e, t.type, fn(t.defaultValue)), t.checked == null && t.defaultChecked != null && (e.defaultChecked = !!t.defaultChecked);
}
function Dc(e, t, n) {
	if (t.hasOwnProperty('value') || t.hasOwnProperty('defaultValue')) {
		var r = t.type;
		if (!((r !== 'submit' && r !== 'reset') || (t.value !== void 0 && t.value !== null))) return;
		(t = '' + e._wrapperState.initialValue), n || t === e.value || (e.value = t), (e.defaultValue = t);
	}
	(n = e.name), n !== '' && (e.name = ''), (e.defaultChecked = !!e._wrapperState.initialChecked), n !== '' && (e.name = n);
}
function Ql(e, t, n) {
	(t !== 'number' || Di(e.ownerDocument) !== e) && (n == null ? (e.defaultValue = '' + e._wrapperState.initialValue) : e.defaultValue !== '' + n && (e.defaultValue = '' + n));
}
var Jr = Array.isArray;
function dr(e, t, n, r) {
	if (((e = e.options), t)) {
		t = {};
		for (var o = 0; o < n.length; o++) t['$' + n[o]] = !0;
		for (n = 0; n < e.length; n++) (o = t.hasOwnProperty('$' + e[n].value)), e[n].selected !== o && (e[n].selected = o), o && r && (e[n].defaultSelected = !0);
	} else {
		for (n = '' + fn(n), t = null, o = 0; o < e.length; o++) {
			if (e[o].value === n) {
				(e[o].selected = !0), r && (e[o].defaultSelected = !0);
				return;
			}
			t !== null || e[o].disabled || (t = e[o]);
		}
		t !== null && (t.selected = !0);
	}
}
function Xl(e, t) {
	if (t.dangerouslySetInnerHTML != null) throw Error(_(91));
	return de({}, t, {value: void 0, defaultValue: void 0, children: '' + e._wrapperState.initialValue});
}
function Lc(e, t) {
	var n = t.value;
	if (n == null) {
		if (((n = t.children), (t = t.defaultValue), n != null)) {
			if (t != null) throw Error(_(92));
			if (Jr(n)) {
				if (1 < n.length) throw Error(_(93));
				n = n[0];
			}
			t = n;
		}
		t == null && (t = ''), (n = t);
	}
	e._wrapperState = {initialValue: fn(n)};
}
function Hf(e, t) {
	var n = fn(t.value),
		r = fn(t.defaultValue);
	n != null && ((n = '' + n), n !== e.value && (e.value = n), t.defaultValue == null && e.defaultValue !== n && (e.defaultValue = n)), r != null && (e.defaultValue = '' + r);
}
function Fc(e) {
	var t = e.textContent;
	t === e._wrapperState.initialValue && t !== '' && t !== null && (e.value = t);
}
function Wf(e) {
	switch (e) {
		case 'svg':
			return 'http://www.w3.org/2000/svg';
		case 'math':
			return 'http://www.w3.org/1998/Math/MathML';
		default:
			return 'http://www.w3.org/1999/xhtml';
	}
}
function Zl(e, t) {
	return e == null || e === 'http://www.w3.org/1999/xhtml' ? Wf(t) : e === 'http://www.w3.org/2000/svg' && t === 'foreignObject' ? 'http://www.w3.org/1999/xhtml' : e;
}
var Jo,
	Vf = (function (e) {
		return typeof MSApp < 'u' && MSApp.execUnsafeLocalFunction
			? function (t, n, r, o) {
					MSApp.execUnsafeLocalFunction(function () {
						return e(t, n, r, o);
					});
			  }
			: e;
	})(function (e, t) {
		if (e.namespaceURI !== 'http://www.w3.org/2000/svg' || 'innerHTML' in e) e.innerHTML = t;
		else {
			for (Jo = Jo || document.createElement('div'), Jo.innerHTML = '<svg>' + t.valueOf().toString() + '</svg>', t = Jo.firstChild; e.firstChild; ) e.removeChild(e.firstChild);
			for (; t.firstChild; ) e.appendChild(t.firstChild);
		}
	});
function mo(e, t) {
	if (t) {
		var n = e.firstChild;
		if (n && n === e.lastChild && n.nodeType === 3) {
			n.nodeValue = t;
			return;
		}
	}
	e.textContent = t;
}
var ro = {
		animationIterationCount: !0,
		aspectRatio: !0,
		borderImageOutset: !0,
		borderImageSlice: !0,
		borderImageWidth: !0,
		boxFlex: !0,
		boxFlexGroup: !0,
		boxOrdinalGroup: !0,
		columnCount: !0,
		columns: !0,
		flex: !0,
		flexGrow: !0,
		flexPositive: !0,
		flexShrink: !0,
		flexNegative: !0,
		flexOrder: !0,
		gridArea: !0,
		gridRow: !0,
		gridRowEnd: !0,
		gridRowSpan: !0,
		gridRowStart: !0,
		gridColumn: !0,
		gridColumnEnd: !0,
		gridColumnSpan: !0,
		gridColumnStart: !0,
		fontWeight: !0,
		lineClamp: !0,
		lineHeight: !0,
		opacity: !0,
		order: !0,
		orphans: !0,
		tabSize: !0,
		widows: !0,
		zIndex: !0,
		zoom: !0,
		fillOpacity: !0,
		floodOpacity: !0,
		stopOpacity: !0,
		strokeDasharray: !0,
		strokeDashoffset: !0,
		strokeMiterlimit: !0,
		strokeOpacity: !0,
		strokeWidth: !0,
	},
	py = ['Webkit', 'ms', 'Moz', 'O'];
Object.keys(ro).forEach(function (e) {
	py.forEach(function (t) {
		(t = t + e.charAt(0).toUpperCase() + e.substring(1)), (ro[t] = ro[e]);
	});
});
function Gf(e, t, n) {
	return t == null || typeof t == 'boolean' || t === '' ? '' : n || typeof t != 'number' || t === 0 || (ro.hasOwnProperty(e) && ro[e]) ? ('' + t).trim() : t + 'px';
}
function Kf(e, t) {
	e = e.style;
	for (var n in t)
		if (t.hasOwnProperty(n)) {
			var r = n.indexOf('--') === 0,
				o = Gf(n, t[n], r);
			n === 'float' && (n = 'cssFloat'), r ? e.setProperty(n, o) : (e[n] = o);
		}
}
var my = de({menuitem: !0}, {area: !0, base: !0, br: !0, col: !0, embed: !0, hr: !0, img: !0, input: !0, keygen: !0, link: !0, meta: !0, param: !0, source: !0, track: !0, wbr: !0});
function ql(e, t) {
	if (t) {
		if (my[e] && (t.children != null || t.dangerouslySetInnerHTML != null)) throw Error(_(137, e));
		if (t.dangerouslySetInnerHTML != null) {
			if (t.children != null) throw Error(_(60));
			if (typeof t.dangerouslySetInnerHTML != 'object' || !('__html' in t.dangerouslySetInnerHTML)) throw Error(_(61));
		}
		if (t.style != null && typeof t.style != 'object') throw Error(_(62));
	}
}
function Jl(e, t) {
	if (e.indexOf('-') === -1) return typeof t.is == 'string';
	switch (e) {
		case 'annotation-xml':
		case 'color-profile':
		case 'font-face':
		case 'font-face-src':
		case 'font-face-uri':
		case 'font-face-format':
		case 'font-face-name':
		case 'missing-glyph':
			return !1;
		default:
			return !0;
	}
}
var ea = null;
function uu(e) {
	return (e = e.target || e.srcElement || window), e.correspondingUseElement && (e = e.correspondingUseElement), e.nodeType === 3 ? e.parentNode : e;
}
var ta = null,
	fr = null,
	pr = null;
function zc(e) {
	if ((e = $o(e))) {
		if (typeof ta != 'function') throw Error(_(280));
		var t = e.stateNode;
		t && ((t = xs(t)), ta(e.stateNode, e.type, t));
	}
}
function Yf(e) {
	fr ? (pr ? pr.push(e) : (pr = [e])) : (fr = e);
}
function Qf() {
	if (fr) {
		var e = fr,
			t = pr;
		if (((pr = fr = null), zc(e), t)) for (e = 0; e < t.length; e++) zc(t[e]);
	}
}
function Xf(e, t) {
	return e(t);
}
function Zf() {}
var sl = !1;
function qf(e, t, n) {
	if (sl) return e(t, n);
	sl = !0;
	try {
		return Xf(e, t, n);
	} finally {
		(sl = !1), (fr !== null || pr !== null) && (Zf(), Qf());
	}
}
function ho(e, t) {
	var n = e.stateNode;
	if (n === null) return null;
	var r = xs(n);
	if (r === null) return null;
	n = r[t];
	e: switch (t) {
		case 'onClick':
		case 'onClickCapture':
		case 'onDoubleClick':
		case 'onDoubleClickCapture':
		case 'onMouseDown':
		case 'onMouseDownCapture':
		case 'onMouseMove':
		case 'onMouseMoveCapture':
		case 'onMouseUp':
		case 'onMouseUpCapture':
		case 'onMouseEnter':
			(r = !r.disabled) || ((e = e.type), (r = !(e === 'button' || e === 'input' || e === 'select' || e === 'textarea'))), (e = !r);
			break e;
		default:
			e = !1;
	}
	if (e) return null;
	if (n && typeof n != 'function') throw Error(_(231, t, typeof n));
	return n;
}
var na = !1;
if (Lt)
	try {
		var $r = {};
		Object.defineProperty($r, 'passive', {
			get: function () {
				na = !0;
			},
		}),
			window.addEventListener('test', $r, $r),
			window.removeEventListener('test', $r, $r);
	} catch {
		na = !1;
	}
function hy(e, t, n, r, o, i, s, l, a) {
	var u = Array.prototype.slice.call(arguments, 3);
	try {
		t.apply(n, u);
	} catch (f) {
		this.onError(f);
	}
}
var oo = !1,
	Li = null,
	Fi = !1,
	ra = null,
	gy = {
		onError: function (e) {
			(oo = !0), (Li = e);
		},
	};
function vy(e, t, n, r, o, i, s, l, a) {
	(oo = !1), (Li = null), hy.apply(gy, arguments);
}
function yy(e, t, n, r, o, i, s, l, a) {
	if ((vy.apply(this, arguments), oo)) {
		if (oo) {
			var u = Li;
			(oo = !1), (Li = null);
		} else throw Error(_(198));
		Fi || ((Fi = !0), (ra = u));
	}
}
function Bn(e) {
	var t = e,
		n = e;
	if (e.alternate) for (; t.return; ) t = t.return;
	else {
		e = t;
		do (t = e), t.flags & 4098 && (n = t.return), (e = t.return);
		while (e);
	}
	return t.tag === 3 ? n : null;
}
function Jf(e) {
	if (e.tag === 13) {
		var t = e.memoizedState;
		if ((t === null && ((e = e.alternate), e !== null && (t = e.memoizedState)), t !== null)) return t.dehydrated;
	}
	return null;
}
function $c(e) {
	if (Bn(e) !== e) throw Error(_(188));
}
function wy(e) {
	var t = e.alternate;
	if (!t) {
		if (((t = Bn(e)), t === null)) throw Error(_(188));
		return t !== e ? null : e;
	}
	for (var n = e, r = t; ; ) {
		var o = n.return;
		if (o === null) break;
		var i = o.alternate;
		if (i === null) {
			if (((r = o.return), r !== null)) {
				n = r;
				continue;
			}
			break;
		}
		if (o.child === i.child) {
			for (i = o.child; i; ) {
				if (i === n) return $c(o), e;
				if (i === r) return $c(o), t;
				i = i.sibling;
			}
			throw Error(_(188));
		}
		if (n.return !== r.return) (n = o), (r = i);
		else {
			for (var s = !1, l = o.child; l; ) {
				if (l === n) {
					(s = !0), (n = o), (r = i);
					break;
				}
				if (l === r) {
					(s = !0), (r = o), (n = i);
					break;
				}
				l = l.sibling;
			}
			if (!s) {
				for (l = i.child; l; ) {
					if (l === n) {
						(s = !0), (n = i), (r = o);
						break;
					}
					if (l === r) {
						(s = !0), (r = i), (n = o);
						break;
					}
					l = l.sibling;
				}
				if (!s) throw Error(_(189));
			}
		}
		if (n.alternate !== r) throw Error(_(190));
	}
	if (n.tag !== 3) throw Error(_(188));
	return n.stateNode.current === n ? e : t;
}
function ep(e) {
	return (e = wy(e)), e !== null ? tp(e) : null;
}
function tp(e) {
	if (e.tag === 5 || e.tag === 6) return e;
	for (e = e.child; e !== null; ) {
		var t = tp(e);
		if (t !== null) return t;
		e = e.sibling;
	}
	return null;
}
var np = Xe.unstable_scheduleCallback,
	Bc = Xe.unstable_cancelCallback,
	xy = Xe.unstable_shouldYield,
	Sy = Xe.unstable_requestPaint,
	me = Xe.unstable_now,
	Ey = Xe.unstable_getCurrentPriorityLevel,
	cu = Xe.unstable_ImmediatePriority,
	rp = Xe.unstable_UserBlockingPriority,
	zi = Xe.unstable_NormalPriority,
	Cy = Xe.unstable_LowPriority,
	op = Xe.unstable_IdlePriority,
	gs = null,
	Tt = null;
function by(e) {
	if (Tt && typeof Tt.onCommitFiberRoot == 'function')
		try {
			Tt.onCommitFiberRoot(gs, e, void 0, (e.current.flags & 128) === 128);
		} catch {}
}
var pt = Math.clz32 ? Math.clz32 : Py,
	ky = Math.log,
	Ty = Math.LN2;
function Py(e) {
	return (e >>>= 0), e === 0 ? 32 : (31 - ((ky(e) / Ty) | 0)) | 0;
}
var ei = 64,
	ti = 4194304;
function eo(e) {
	switch (e & -e) {
		case 1:
			return 1;
		case 2:
			return 2;
		case 4:
			return 4;
		case 8:
			return 8;
		case 16:
			return 16;
		case 32:
			return 32;
		case 64:
		case 128:
		case 256:
		case 512:
		case 1024:
		case 2048:
		case 4096:
		case 8192:
		case 16384:
		case 32768:
		case 65536:
		case 131072:
		case 262144:
		case 524288:
		case 1048576:
		case 2097152:
			return e & 4194240;
		case 4194304:
		case 8388608:
		case 16777216:
		case 33554432:
		case 67108864:
			return e & 130023424;
		case 134217728:
			return 134217728;
		case 268435456:
			return 268435456;
		case 536870912:
			return 536870912;
		case 1073741824:
			return 1073741824;
		default:
			return e;
	}
}
function $i(e, t) {
	var n = e.pendingLanes;
	if (n === 0) return 0;
	var r = 0,
		o = e.suspendedLanes,
		i = e.pingedLanes,
		s = n & 268435455;
	if (s !== 0) {
		var l = s & ~o;
		l !== 0 ? (r = eo(l)) : ((i &= s), i !== 0 && (r = eo(i)));
	} else (s = n & ~o), s !== 0 ? (r = eo(s)) : i !== 0 && (r = eo(i));
	if (r === 0) return 0;
	if (t !== 0 && t !== r && !(t & o) && ((o = r & -r), (i = t & -t), o >= i || (o === 16 && (i & 4194240) !== 0))) return t;
	if ((r & 4 && (r |= n & 16), (t = e.entangledLanes), t !== 0)) for (e = e.entanglements, t &= r; 0 < t; ) (n = 31 - pt(t)), (o = 1 << n), (r |= e[n]), (t &= ~o);
	return r;
}
function Ny(e, t) {
	switch (e) {
		case 1:
		case 2:
		case 4:
			return t + 250;
		case 8:
		case 16:
		case 32:
		case 64:
		case 128:
		case 256:
		case 512:
		case 1024:
		case 2048:
		case 4096:
		case 8192:
		case 16384:
		case 32768:
		case 65536:
		case 131072:
		case 262144:
		case 524288:
		case 1048576:
		case 2097152:
			return t + 5e3;
		case 4194304:
		case 8388608:
		case 16777216:
		case 33554432:
		case 67108864:
			return -1;
		case 134217728:
		case 268435456:
		case 536870912:
		case 1073741824:
			return -1;
		default:
			return -1;
	}
}
function _y(e, t) {
	for (var n = e.suspendedLanes, r = e.pingedLanes, o = e.expirationTimes, i = e.pendingLanes; 0 < i; ) {
		var s = 31 - pt(i),
			l = 1 << s,
			a = o[s];
		a === -1 ? (!(l & n) || l & r) && (o[s] = Ny(l, t)) : a <= t && (e.expiredLanes |= l), (i &= ~l);
	}
}
function oa(e) {
	return (e = e.pendingLanes & -1073741825), e !== 0 ? e : e & 1073741824 ? 1073741824 : 0;
}
function ip() {
	var e = ei;
	return (ei <<= 1), !(ei & 4194240) && (ei = 64), e;
}
function ll(e) {
	for (var t = [], n = 0; 31 > n; n++) t.push(e);
	return t;
}
function Fo(e, t, n) {
	(e.pendingLanes |= t), t !== 536870912 && ((e.suspendedLanes = 0), (e.pingedLanes = 0)), (e = e.eventTimes), (t = 31 - pt(t)), (e[t] = n);
}
function Ry(e, t) {
	var n = e.pendingLanes & ~t;
	(e.pendingLanes = t), (e.suspendedLanes = 0), (e.pingedLanes = 0), (e.expiredLanes &= t), (e.mutableReadLanes &= t), (e.entangledLanes &= t), (t = e.entanglements);
	var r = e.eventTimes;
	for (e = e.expirationTimes; 0 < n; ) {
		var o = 31 - pt(n),
			i = 1 << o;
		(t[o] = 0), (r[o] = -1), (e[o] = -1), (n &= ~i);
	}
}
function du(e, t) {
	var n = (e.entangledLanes |= t);
	for (e = e.entanglements; n; ) {
		var r = 31 - pt(n),
			o = 1 << r;
		(o & t) | (e[r] & t) && (e[r] |= t), (n &= ~o);
	}
}
var q = 0;
function sp(e) {
	return (e &= -e), 1 < e ? (4 < e ? (e & 268435455 ? 16 : 536870912) : 4) : 1;
}
var lp,
	fu,
	ap,
	up,
	cp,
	ia = !1,
	ni = [],
	rn = null,
	on = null,
	sn = null,
	go = new Map(),
	vo = new Map(),
	Jt = [],
	Ay =
		'mousedown mouseup touchcancel touchend touchstart auxclick dblclick pointercancel pointerdown pointerup dragend dragstart drop compositionend compositionstart keydown keypress keyup input textInput copy cut paste click change contextmenu reset submit'.split(
			' '
		);
function Uc(e, t) {
	switch (e) {
		case 'focusin':
		case 'focusout':
			rn = null;
			break;
		case 'dragenter':
		case 'dragleave':
			on = null;
			break;
		case 'mouseover':
		case 'mouseout':
			sn = null;
			break;
		case 'pointerover':
		case 'pointerout':
			go.delete(t.pointerId);
			break;
		case 'gotpointercapture':
		case 'lostpointercapture':
			vo.delete(t.pointerId);
	}
}
function Br(e, t, n, r, o, i) {
	return e === null || e.nativeEvent !== i
		? ((e = {blockedOn: t, domEventName: n, eventSystemFlags: r, nativeEvent: i, targetContainers: [o]}), t !== null && ((t = $o(t)), t !== null && fu(t)), e)
		: ((e.eventSystemFlags |= r), (t = e.targetContainers), o !== null && t.indexOf(o) === -1 && t.push(o), e);
}
function My(e, t, n, r, o) {
	switch (t) {
		case 'focusin':
			return (rn = Br(rn, e, t, n, r, o)), !0;
		case 'dragenter':
			return (on = Br(on, e, t, n, r, o)), !0;
		case 'mouseover':
			return (sn = Br(sn, e, t, n, r, o)), !0;
		case 'pointerover':
			var i = o.pointerId;
			return go.set(i, Br(go.get(i) || null, e, t, n, r, o)), !0;
		case 'gotpointercapture':
			return (i = o.pointerId), vo.set(i, Br(vo.get(i) || null, e, t, n, r, o)), !0;
	}
	return !1;
}
function dp(e) {
	var t = bn(e.target);
	if (t !== null) {
		var n = Bn(t);
		if (n !== null) {
			if (((t = n.tag), t === 13)) {
				if (((t = Jf(n)), t !== null)) {
					(e.blockedOn = t),
						cp(e.priority, function () {
							ap(n);
						});
					return;
				}
			} else if (t === 3 && n.stateNode.current.memoizedState.isDehydrated) {
				e.blockedOn = n.tag === 3 ? n.stateNode.containerInfo : null;
				return;
			}
		}
	}
	e.blockedOn = null;
}
function Ei(e) {
	if (e.blockedOn !== null) return !1;
	for (var t = e.targetContainers; 0 < t.length; ) {
		var n = sa(e.domEventName, e.eventSystemFlags, t[0], e.nativeEvent);
		if (n === null) {
			n = e.nativeEvent;
			var r = new n.constructor(n.type, n);
			(ea = r), n.target.dispatchEvent(r), (ea = null);
		} else return (t = $o(n)), t !== null && fu(t), (e.blockedOn = n), !1;
		t.shift();
	}
	return !0;
}
function Hc(e, t, n) {
	Ei(e) && n.delete(t);
}
function Oy() {
	(ia = !1), rn !== null && Ei(rn) && (rn = null), on !== null && Ei(on) && (on = null), sn !== null && Ei(sn) && (sn = null), go.forEach(Hc), vo.forEach(Hc);
}
function Ur(e, t) {
	e.blockedOn === t && ((e.blockedOn = null), ia || ((ia = !0), Xe.unstable_scheduleCallback(Xe.unstable_NormalPriority, Oy)));
}
function yo(e) {
	function t(o) {
		return Ur(o, e);
	}
	if (0 < ni.length) {
		Ur(ni[0], e);
		for (var n = 1; n < ni.length; n++) {
			var r = ni[n];
			r.blockedOn === e && (r.blockedOn = null);
		}
	}
	for (rn !== null && Ur(rn, e), on !== null && Ur(on, e), sn !== null && Ur(sn, e), go.forEach(t), vo.forEach(t), n = 0; n < Jt.length; n++) (r = Jt[n]), r.blockedOn === e && (r.blockedOn = null);
	for (; 0 < Jt.length && ((n = Jt[0]), n.blockedOn === null); ) dp(n), n.blockedOn === null && Jt.shift();
}
var mr = Ht.ReactCurrentBatchConfig,
	Bi = !0;
function jy(e, t, n, r) {
	var o = q,
		i = mr.transition;
	mr.transition = null;
	try {
		(q = 1), pu(e, t, n, r);
	} finally {
		(q = o), (mr.transition = i);
	}
}
function Iy(e, t, n, r) {
	var o = q,
		i = mr.transition;
	mr.transition = null;
	try {
		(q = 4), pu(e, t, n, r);
	} finally {
		(q = o), (mr.transition = i);
	}
}
function pu(e, t, n, r) {
	if (Bi) {
		var o = sa(e, t, n, r);
		if (o === null) vl(e, t, r, Ui, n), Uc(e, r);
		else if (My(o, e, t, n, r)) r.stopPropagation();
		else if ((Uc(e, r), t & 4 && -1 < Ay.indexOf(e))) {
			for (; o !== null; ) {
				var i = $o(o);
				if ((i !== null && lp(i), (i = sa(e, t, n, r)), i === null && vl(e, t, r, Ui, n), i === o)) break;
				o = i;
			}
			o !== null && r.stopPropagation();
		} else vl(e, t, r, null, n);
	}
}
var Ui = null;
function sa(e, t, n, r) {
	if (((Ui = null), (e = uu(r)), (e = bn(e)), e !== null))
		if (((t = Bn(e)), t === null)) e = null;
		else if (((n = t.tag), n === 13)) {
			if (((e = Jf(t)), e !== null)) return e;
			e = null;
		} else if (n === 3) {
			if (t.stateNode.current.memoizedState.isDehydrated) return t.tag === 3 ? t.stateNode.containerInfo : null;
			e = null;
		} else t !== e && (e = null);
	return (Ui = e), null;
}
function fp(e) {
	switch (e) {
		case 'cancel':
		case 'click':
		case 'close':
		case 'contextmenu':
		case 'copy':
		case 'cut':
		case 'auxclick':
		case 'dblclick':
		case 'dragend':
		case 'dragstart':
		case 'drop':
		case 'focusin':
		case 'focusout':
		case 'input':
		case 'invalid':
		case 'keydown':
		case 'keypress':
		case 'keyup':
		case 'mousedown':
		case 'mouseup':
		case 'paste':
		case 'pause':
		case 'play':
		case 'pointercancel':
		case 'pointerdown':
		case 'pointerup':
		case 'ratechange':
		case 'reset':
		case 'resize':
		case 'seeked':
		case 'submit':
		case 'touchcancel':
		case 'touchend':
		case 'touchstart':
		case 'volumechange':
		case 'change':
		case 'selectionchange':
		case 'textInput':
		case 'compositionstart':
		case 'compositionend':
		case 'compositionupdate':
		case 'beforeblur':
		case 'afterblur':
		case 'beforeinput':
		case 'blur':
		case 'fullscreenchange':
		case 'focus':
		case 'hashchange':
		case 'popstate':
		case 'select':
		case 'selectstart':
			return 1;
		case 'drag':
		case 'dragenter':
		case 'dragexit':
		case 'dragleave':
		case 'dragover':
		case 'mousemove':
		case 'mouseout':
		case 'mouseover':
		case 'pointermove':
		case 'pointerout':
		case 'pointerover':
		case 'scroll':
		case 'toggle':
		case 'touchmove':
		case 'wheel':
		case 'mouseenter':
		case 'mouseleave':
		case 'pointerenter':
		case 'pointerleave':
			return 4;
		case 'message':
			switch (Ey()) {
				case cu:
					return 1;
				case rp:
					return 4;
				case zi:
				case Cy:
					return 16;
				case op:
					return 536870912;
				default:
					return 16;
			}
		default:
			return 16;
	}
}
var tn = null,
	mu = null,
	Ci = null;
function pp() {
	if (Ci) return Ci;
	var e,
		t = mu,
		n = t.length,
		r,
		o = 'value' in tn ? tn.value : tn.textContent,
		i = o.length;
	for (e = 0; e < n && t[e] === o[e]; e++);
	var s = n - e;
	for (r = 1; r <= s && t[n - r] === o[i - r]; r++);
	return (Ci = o.slice(e, 1 < r ? 1 - r : void 0));
}
function bi(e) {
	var t = e.keyCode;
	return 'charCode' in e ? ((e = e.charCode), e === 0 && t === 13 && (e = 13)) : (e = t), e === 10 && (e = 13), 32 <= e || e === 13 ? e : 0;
}
function ri() {
	return !0;
}
function Wc() {
	return !1;
}
function qe(e) {
	function t(n, r, o, i, s) {
		(this._reactName = n), (this._targetInst = o), (this.type = r), (this.nativeEvent = i), (this.target = s), (this.currentTarget = null);
		for (var l in e) e.hasOwnProperty(l) && ((n = e[l]), (this[l] = n ? n(i) : i[l]));
		return (this.isDefaultPrevented = (i.defaultPrevented != null ? i.defaultPrevented : i.returnValue === !1) ? ri : Wc), (this.isPropagationStopped = Wc), this;
	}
	return (
		de(t.prototype, {
			preventDefault: function () {
				this.defaultPrevented = !0;
				var n = this.nativeEvent;
				n && (n.preventDefault ? n.preventDefault() : typeof n.returnValue != 'unknown' && (n.returnValue = !1), (this.isDefaultPrevented = ri));
			},
			stopPropagation: function () {
				var n = this.nativeEvent;
				n && (n.stopPropagation ? n.stopPropagation() : typeof n.cancelBubble != 'unknown' && (n.cancelBubble = !0), (this.isPropagationStopped = ri));
			},
			persist: function () {},
			isPersistent: ri,
		}),
		t
	);
}
var Or = {
		eventPhase: 0,
		bubbles: 0,
		cancelable: 0,
		timeStamp: function (e) {
			return e.timeStamp || Date.now();
		},
		defaultPrevented: 0,
		isTrusted: 0,
	},
	hu = qe(Or),
	zo = de({}, Or, {view: 0, detail: 0}),
	Dy = qe(zo),
	al,
	ul,
	Hr,
	vs = de({}, zo, {
		screenX: 0,
		screenY: 0,
		clientX: 0,
		clientY: 0,
		pageX: 0,
		pageY: 0,
		ctrlKey: 0,
		shiftKey: 0,
		altKey: 0,
		metaKey: 0,
		getModifierState: gu,
		button: 0,
		buttons: 0,
		relatedTarget: function (e) {
			return e.relatedTarget === void 0 ? (e.fromElement === e.srcElement ? e.toElement : e.fromElement) : e.relatedTarget;
		},
		movementX: function (e) {
			return 'movementX' in e ? e.movementX : (e !== Hr && (Hr && e.type === 'mousemove' ? ((al = e.screenX - Hr.screenX), (ul = e.screenY - Hr.screenY)) : (ul = al = 0), (Hr = e)), al);
		},
		movementY: function (e) {
			return 'movementY' in e ? e.movementY : ul;
		},
	}),
	Vc = qe(vs),
	Ly = de({}, vs, {dataTransfer: 0}),
	Fy = qe(Ly),
	zy = de({}, zo, {relatedTarget: 0}),
	cl = qe(zy),
	$y = de({}, Or, {animationName: 0, elapsedTime: 0, pseudoElement: 0}),
	By = qe($y),
	Uy = de({}, Or, {
		clipboardData: function (e) {
			return 'clipboardData' in e ? e.clipboardData : window.clipboardData;
		},
	}),
	Hy = qe(Uy),
	Wy = de({}, Or, {data: 0}),
	Gc = qe(Wy),
	Vy = {Esc: 'Escape', Spacebar: ' ', Left: 'ArrowLeft', Up: 'ArrowUp', Right: 'ArrowRight', Down: 'ArrowDown', Del: 'Delete', Win: 'OS', Menu: 'ContextMenu', Apps: 'ContextMenu', Scroll: 'ScrollLock', MozPrintableKey: 'Unidentified'},
	Gy = {
		8: 'Backspace',
		9: 'Tab',
		12: 'Clear',
		13: 'Enter',
		16: 'Shift',
		17: 'Control',
		18: 'Alt',
		19: 'Pause',
		20: 'CapsLock',
		27: 'Escape',
		32: ' ',
		33: 'PageUp',
		34: 'PageDown',
		35: 'End',
		36: 'Home',
		37: 'ArrowLeft',
		38: 'ArrowUp',
		39: 'ArrowRight',
		40: 'ArrowDown',
		45: 'Insert',
		46: 'Delete',
		112: 'F1',
		113: 'F2',
		114: 'F3',
		115: 'F4',
		116: 'F5',
		117: 'F6',
		118: 'F7',
		119: 'F8',
		120: 'F9',
		121: 'F10',
		122: 'F11',
		123: 'F12',
		144: 'NumLock',
		145: 'ScrollLock',
		224: 'Meta',
	},
	Ky = {Alt: 'altKey', Control: 'ctrlKey', Meta: 'metaKey', Shift: 'shiftKey'};
function Yy(e) {
	var t = this.nativeEvent;
	return t.getModifierState ? t.getModifierState(e) : (e = Ky[e]) ? !!t[e] : !1;
}
function gu() {
	return Yy;
}
var Qy = de({}, zo, {
		key: function (e) {
			if (e.key) {
				var t = Vy[e.key] || e.key;
				if (t !== 'Unidentified') return t;
			}
			return e.type === 'keypress' ? ((e = bi(e)), e === 13 ? 'Enter' : String.fromCharCode(e)) : e.type === 'keydown' || e.type === 'keyup' ? Gy[e.keyCode] || 'Unidentified' : '';
		},
		code: 0,
		location: 0,
		ctrlKey: 0,
		shiftKey: 0,
		altKey: 0,
		metaKey: 0,
		repeat: 0,
		locale: 0,
		getModifierState: gu,
		charCode: function (e) {
			return e.type === 'keypress' ? bi(e) : 0;
		},
		keyCode: function (e) {
			return e.type === 'keydown' || e.type === 'keyup' ? e.keyCode : 0;
		},
		which: function (e) {
			return e.type === 'keypress' ? bi(e) : e.type === 'keydown' || e.type === 'keyup' ? e.keyCode : 0;
		},
	}),
	Xy = qe(Qy),
	Zy = de({}, vs, {pointerId: 0, width: 0, height: 0, pressure: 0, tangentialPressure: 0, tiltX: 0, tiltY: 0, twist: 0, pointerType: 0, isPrimary: 0}),
	Kc = qe(Zy),
	qy = de({}, zo, {touches: 0, targetTouches: 0, changedTouches: 0, altKey: 0, metaKey: 0, ctrlKey: 0, shiftKey: 0, getModifierState: gu}),
	Jy = qe(qy),
	e0 = de({}, Or, {propertyName: 0, elapsedTime: 0, pseudoElement: 0}),
	t0 = qe(e0),
	n0 = de({}, vs, {
		deltaX: function (e) {
			return 'deltaX' in e ? e.deltaX : 'wheelDeltaX' in e ? -e.wheelDeltaX : 0;
		},
		deltaY: function (e) {
			return 'deltaY' in e ? e.deltaY : 'wheelDeltaY' in e ? -e.wheelDeltaY : 'wheelDelta' in e ? -e.wheelDelta : 0;
		},
		deltaZ: 0,
		deltaMode: 0,
	}),
	r0 = qe(n0),
	o0 = [9, 13, 27, 32],
	vu = Lt && 'CompositionEvent' in window,
	io = null;
Lt && 'documentMode' in document && (io = document.documentMode);
var i0 = Lt && 'TextEvent' in window && !io,
	mp = Lt && (!vu || (io && 8 < io && 11 >= io)),
	Yc = ' ',
	Qc = !1;
function hp(e, t) {
	switch (e) {
		case 'keyup':
			return o0.indexOf(t.keyCode) !== -1;
		case 'keydown':
			return t.keyCode !== 229;
		case 'keypress':
		case 'mousedown':
		case 'focusout':
			return !0;
		default:
			return !1;
	}
}
function gp(e) {
	return (e = e.detail), typeof e == 'object' && 'data' in e ? e.data : null;
}
var tr = !1;
function s0(e, t) {
	switch (e) {
		case 'compositionend':
			return gp(t);
		case 'keypress':
			return t.which !== 32 ? null : ((Qc = !0), Yc);
		case 'textInput':
			return (e = t.data), e === Yc && Qc ? null : e;
		default:
			return null;
	}
}
function l0(e, t) {
	if (tr) return e === 'compositionend' || (!vu && hp(e, t)) ? ((e = pp()), (Ci = mu = tn = null), (tr = !1), e) : null;
	switch (e) {
		case 'paste':
			return null;
		case 'keypress':
			if (!(t.ctrlKey || t.altKey || t.metaKey) || (t.ctrlKey && t.altKey)) {
				if (t.char && 1 < t.char.length) return t.char;
				if (t.which) return String.fromCharCode(t.which);
			}
			return null;
		case 'compositionend':
			return mp && t.locale !== 'ko' ? null : t.data;
		default:
			return null;
	}
}
var a0 = {color: !0, date: !0, datetime: !0, 'datetime-local': !0, email: !0, month: !0, number: !0, password: !0, range: !0, search: !0, tel: !0, text: !0, time: !0, url: !0, week: !0};
function Xc(e) {
	var t = e && e.nodeName && e.nodeName.toLowerCase();
	return t === 'input' ? !!a0[e.type] : t === 'textarea';
}
function vp(e, t, n, r) {
	Yf(r), (t = Hi(t, 'onChange')), 0 < t.length && ((n = new hu('onChange', 'change', null, n, r)), e.push({event: n, listeners: t}));
}
var so = null,
	wo = null;
function u0(e) {
	Np(e, 0);
}
function ys(e) {
	var t = or(e);
	if (Bf(t)) return e;
}
function c0(e, t) {
	if (e === 'change') return t;
}
var yp = !1;
if (Lt) {
	var dl;
	if (Lt) {
		var fl = 'oninput' in document;
		if (!fl) {
			var Zc = document.createElement('div');
			Zc.setAttribute('oninput', 'return;'), (fl = typeof Zc.oninput == 'function');
		}
		dl = fl;
	} else dl = !1;
	yp = dl && (!document.documentMode || 9 < document.documentMode);
}
function qc() {
	so && (so.detachEvent('onpropertychange', wp), (wo = so = null));
}
function wp(e) {
	if (e.propertyName === 'value' && ys(wo)) {
		var t = [];
		vp(t, wo, e, uu(e)), qf(u0, t);
	}
}
function d0(e, t, n) {
	e === 'focusin' ? (qc(), (so = t), (wo = n), so.attachEvent('onpropertychange', wp)) : e === 'focusout' && qc();
}
function f0(e) {
	if (e === 'selectionchange' || e === 'keyup' || e === 'keydown') return ys(wo);
}
function p0(e, t) {
	if (e === 'click') return ys(t);
}
function m0(e, t) {
	if (e === 'input' || e === 'change') return ys(t);
}
function h0(e, t) {
	return (e === t && (e !== 0 || 1 / e === 1 / t)) || (e !== e && t !== t);
}
var ht = typeof Object.is == 'function' ? Object.is : h0;
function xo(e, t) {
	if (ht(e, t)) return !0;
	if (typeof e != 'object' || e === null || typeof t != 'object' || t === null) return !1;
	var n = Object.keys(e),
		r = Object.keys(t);
	if (n.length !== r.length) return !1;
	for (r = 0; r < n.length; r++) {
		var o = n[r];
		if (!Ul.call(t, o) || !ht(e[o], t[o])) return !1;
	}
	return !0;
}
function Jc(e) {
	for (; e && e.firstChild; ) e = e.firstChild;
	return e;
}
function ed(e, t) {
	var n = Jc(e);
	e = 0;
	for (var r; n; ) {
		if (n.nodeType === 3) {
			if (((r = e + n.textContent.length), e <= t && r >= t)) return {node: n, offset: t - e};
			e = r;
		}
		e: {
			for (; n; ) {
				if (n.nextSibling) {
					n = n.nextSibling;
					break e;
				}
				n = n.parentNode;
			}
			n = void 0;
		}
		n = Jc(n);
	}
}
function xp(e, t) {
	return e && t ? (e === t ? !0 : e && e.nodeType === 3 ? !1 : t && t.nodeType === 3 ? xp(e, t.parentNode) : 'contains' in e ? e.contains(t) : e.compareDocumentPosition ? !!(e.compareDocumentPosition(t) & 16) : !1) : !1;
}
function Sp() {
	for (var e = window, t = Di(); t instanceof e.HTMLIFrameElement; ) {
		try {
			var n = typeof t.contentWindow.location.href == 'string';
		} catch {
			n = !1;
		}
		if (n) e = t.contentWindow;
		else break;
		t = Di(e.document);
	}
	return t;
}
function yu(e) {
	var t = e && e.nodeName && e.nodeName.toLowerCase();
	return t && ((t === 'input' && (e.type === 'text' || e.type === 'search' || e.type === 'tel' || e.type === 'url' || e.type === 'password')) || t === 'textarea' || e.contentEditable === 'true');
}
function g0(e) {
	var t = Sp(),
		n = e.focusedElem,
		r = e.selectionRange;
	if (t !== n && n && n.ownerDocument && xp(n.ownerDocument.documentElement, n)) {
		if (r !== null && yu(n)) {
			if (((t = r.start), (e = r.end), e === void 0 && (e = t), 'selectionStart' in n)) (n.selectionStart = t), (n.selectionEnd = Math.min(e, n.value.length));
			else if (((e = ((t = n.ownerDocument || document) && t.defaultView) || window), e.getSelection)) {
				e = e.getSelection();
				var o = n.textContent.length,
					i = Math.min(r.start, o);
				(r = r.end === void 0 ? i : Math.min(r.end, o)), !e.extend && i > r && ((o = r), (r = i), (i = o)), (o = ed(n, i));
				var s = ed(n, r);
				o &&
					s &&
					(e.rangeCount !== 1 || e.anchorNode !== o.node || e.anchorOffset !== o.offset || e.focusNode !== s.node || e.focusOffset !== s.offset) &&
					((t = t.createRange()), t.setStart(o.node, o.offset), e.removeAllRanges(), i > r ? (e.addRange(t), e.extend(s.node, s.offset)) : (t.setEnd(s.node, s.offset), e.addRange(t)));
			}
		}
		for (t = [], e = n; (e = e.parentNode); ) e.nodeType === 1 && t.push({element: e, left: e.scrollLeft, top: e.scrollTop});
		for (typeof n.focus == 'function' && n.focus(), n = 0; n < t.length; n++) (e = t[n]), (e.element.scrollLeft = e.left), (e.element.scrollTop = e.top);
	}
}
var v0 = Lt && 'documentMode' in document && 11 >= document.documentMode,
	nr = null,
	la = null,
	lo = null,
	aa = !1;
function td(e, t, n) {
	var r = n.window === n ? n.document : n.nodeType === 9 ? n : n.ownerDocument;
	aa ||
		nr == null ||
		nr !== Di(r) ||
		((r = nr),
		'selectionStart' in r && yu(r)
			? (r = {start: r.selectionStart, end: r.selectionEnd})
			: ((r = ((r.ownerDocument && r.ownerDocument.defaultView) || window).getSelection()), (r = {anchorNode: r.anchorNode, anchorOffset: r.anchorOffset, focusNode: r.focusNode, focusOffset: r.focusOffset})),
		(lo && xo(lo, r)) || ((lo = r), (r = Hi(la, 'onSelect')), 0 < r.length && ((t = new hu('onSelect', 'select', null, t, n)), e.push({event: t, listeners: r}), (t.target = nr))));
}
function oi(e, t) {
	var n = {};
	return (n[e.toLowerCase()] = t.toLowerCase()), (n['Webkit' + e] = 'webkit' + t), (n['Moz' + e] = 'moz' + t), n;
}
var rr = {animationend: oi('Animation', 'AnimationEnd'), animationiteration: oi('Animation', 'AnimationIteration'), animationstart: oi('Animation', 'AnimationStart'), transitionend: oi('Transition', 'TransitionEnd')},
	pl = {},
	Ep = {};
Lt &&
	((Ep = document.createElement('div').style),
	'AnimationEvent' in window || (delete rr.animationend.animation, delete rr.animationiteration.animation, delete rr.animationstart.animation),
	'TransitionEvent' in window || delete rr.transitionend.transition);
function ws(e) {
	if (pl[e]) return pl[e];
	if (!rr[e]) return e;
	var t = rr[e],
		n;
	for (n in t) if (t.hasOwnProperty(n) && n in Ep) return (pl[e] = t[n]);
	return e;
}
var Cp = ws('animationend'),
	bp = ws('animationiteration'),
	kp = ws('animationstart'),
	Tp = ws('transitionend'),
	Pp = new Map(),
	nd =
		'abort auxClick cancel canPlay canPlayThrough click close contextMenu copy cut drag dragEnd dragEnter dragExit dragLeave dragOver dragStart drop durationChange emptied encrypted ended error gotPointerCapture input invalid keyDown keyPress keyUp load loadedData loadedMetadata loadStart lostPointerCapture mouseDown mouseMove mouseOut mouseOver mouseUp paste pause play playing pointerCancel pointerDown pointerMove pointerOut pointerOver pointerUp progress rateChange reset resize seeked seeking stalled submit suspend timeUpdate touchCancel touchEnd touchStart volumeChange scroll toggle touchMove waiting wheel'.split(
			' '
		);
function yn(e, t) {
	Pp.set(e, t), $n(t, [e]);
}
for (var ml = 0; ml < nd.length; ml++) {
	var hl = nd[ml],
		y0 = hl.toLowerCase(),
		w0 = hl[0].toUpperCase() + hl.slice(1);
	yn(y0, 'on' + w0);
}
yn(Cp, 'onAnimationEnd');
yn(bp, 'onAnimationIteration');
yn(kp, 'onAnimationStart');
yn('dblclick', 'onDoubleClick');
yn('focusin', 'onFocus');
yn('focusout', 'onBlur');
yn(Tp, 'onTransitionEnd');
Sr('onMouseEnter', ['mouseout', 'mouseover']);
Sr('onMouseLeave', ['mouseout', 'mouseover']);
Sr('onPointerEnter', ['pointerout', 'pointerover']);
Sr('onPointerLeave', ['pointerout', 'pointerover']);
$n('onChange', 'change click focusin focusout input keydown keyup selectionchange'.split(' '));
$n('onSelect', 'focusout contextmenu dragend focusin keydown keyup mousedown mouseup selectionchange'.split(' '));
$n('onBeforeInput', ['compositionend', 'keypress', 'textInput', 'paste']);
$n('onCompositionEnd', 'compositionend focusout keydown keypress keyup mousedown'.split(' '));
$n('onCompositionStart', 'compositionstart focusout keydown keypress keyup mousedown'.split(' '));
$n('onCompositionUpdate', 'compositionupdate focusout keydown keypress keyup mousedown'.split(' '));
var to = 'abort canplay canplaythrough durationchange emptied encrypted ended error loadeddata loadedmetadata loadstart pause play playing progress ratechange resize seeked seeking stalled suspend timeupdate volumechange waiting'.split(' '),
	x0 = new Set('cancel close invalid load scroll toggle'.split(' ').concat(to));
function rd(e, t, n) {
	var r = e.type || 'unknown-event';
	(e.currentTarget = n), yy(r, t, void 0, e), (e.currentTarget = null);
}
function Np(e, t) {
	t = (t & 4) !== 0;
	for (var n = 0; n < e.length; n++) {
		var r = e[n],
			o = r.event;
		r = r.listeners;
		e: {
			var i = void 0;
			if (t)
				for (var s = r.length - 1; 0 <= s; s--) {
					var l = r[s],
						a = l.instance,
						u = l.currentTarget;
					if (((l = l.listener), a !== i && o.isPropagationStopped())) break e;
					rd(o, l, u), (i = a);
				}
			else
				for (s = 0; s < r.length; s++) {
					if (((l = r[s]), (a = l.instance), (u = l.currentTarget), (l = l.listener), a !== i && o.isPropagationStopped())) break e;
					rd(o, l, u), (i = a);
				}
		}
	}
	if (Fi) throw ((e = ra), (Fi = !1), (ra = null), e);
}
function re(e, t) {
	var n = t[pa];
	n === void 0 && (n = t[pa] = new Set());
	var r = e + '__bubble';
	n.has(r) || (_p(t, e, 2, !1), n.add(r));
}
function gl(e, t, n) {
	var r = 0;
	t && (r |= 4), _p(n, e, r, t);
}
var ii = '_reactListening' + Math.random().toString(36).slice(2);
function So(e) {
	if (!e[ii]) {
		(e[ii] = !0),
			Df.forEach(function (n) {
				n !== 'selectionchange' && (x0.has(n) || gl(n, !1, e), gl(n, !0, e));
			});
		var t = e.nodeType === 9 ? e : e.ownerDocument;
		t === null || t[ii] || ((t[ii] = !0), gl('selectionchange', !1, t));
	}
}
function _p(e, t, n, r) {
	switch (fp(t)) {
		case 1:
			var o = jy;
			break;
		case 4:
			o = Iy;
			break;
		default:
			o = pu;
	}
	(n = o.bind(null, t, n, e)),
		(o = void 0),
		!na || (t !== 'touchstart' && t !== 'touchmove' && t !== 'wheel') || (o = !0),
		r ? (o !== void 0 ? e.addEventListener(t, n, {capture: !0, passive: o}) : e.addEventListener(t, n, !0)) : o !== void 0 ? e.addEventListener(t, n, {passive: o}) : e.addEventListener(t, n, !1);
}
function vl(e, t, n, r, o) {
	var i = r;
	if (!(t & 1) && !(t & 2) && r !== null)
		e: for (;;) {
			if (r === null) return;
			var s = r.tag;
			if (s === 3 || s === 4) {
				var l = r.stateNode.containerInfo;
				if (l === o || (l.nodeType === 8 && l.parentNode === o)) break;
				if (s === 4)
					for (s = r.return; s !== null; ) {
						var a = s.tag;
						if ((a === 3 || a === 4) && ((a = s.stateNode.containerInfo), a === o || (a.nodeType === 8 && a.parentNode === o))) return;
						s = s.return;
					}
				for (; l !== null; ) {
					if (((s = bn(l)), s === null)) return;
					if (((a = s.tag), a === 5 || a === 6)) {
						r = i = s;
						continue e;
					}
					l = l.parentNode;
				}
			}
			r = r.return;
		}
	qf(function () {
		var u = i,
			f = uu(n),
			d = [];
		e: {
			var c = Pp.get(e);
			if (c !== void 0) {
				var g = hu,
					w = e;
				switch (e) {
					case 'keypress':
						if (bi(n) === 0) break e;
					case 'keydown':
					case 'keyup':
						g = Xy;
						break;
					case 'focusin':
						(w = 'focus'), (g = cl);
						break;
					case 'focusout':
						(w = 'blur'), (g = cl);
						break;
					case 'beforeblur':
					case 'afterblur':
						g = cl;
						break;
					case 'click':
						if (n.button === 2) break e;
					case 'auxclick':
					case 'dblclick':
					case 'mousedown':
					case 'mousemove':
					case 'mouseup':
					case 'mouseout':
					case 'mouseover':
					case 'contextmenu':
						g = Vc;
						break;
					case 'drag':
					case 'dragend':
					case 'dragenter':
					case 'dragexit':
					case 'dragleave':
					case 'dragover':
					case 'dragstart':
					case 'drop':
						g = Fy;
						break;
					case 'touchcancel':
					case 'touchend':
					case 'touchmove':
					case 'touchstart':
						g = Jy;
						break;
					case Cp:
					case bp:
					case kp:
						g = By;
						break;
					case Tp:
						g = t0;
						break;
					case 'scroll':
						g = Dy;
						break;
					case 'wheel':
						g = r0;
						break;
					case 'copy':
					case 'cut':
					case 'paste':
						g = Hy;
						break;
					case 'gotpointercapture':
					case 'lostpointercapture':
					case 'pointercancel':
					case 'pointerdown':
					case 'pointermove':
					case 'pointerout':
					case 'pointerover':
					case 'pointerup':
						g = Kc;
				}
				var y = (t & 4) !== 0,
					S = !y && e === 'scroll',
					m = y ? (c !== null ? c + 'Capture' : null) : c;
				y = [];
				for (var h = u, x; h !== null; ) {
					x = h;
					var E = x.stateNode;
					if ((x.tag === 5 && E !== null && ((x = E), m !== null && ((E = ho(h, m)), E != null && y.push(Eo(h, E, x)))), S)) break;
					h = h.return;
				}
				0 < y.length && ((c = new g(c, w, null, n, f)), d.push({event: c, listeners: y}));
			}
		}
		if (!(t & 7)) {
			e: {
				if (((c = e === 'mouseover' || e === 'pointerover'), (g = e === 'mouseout' || e === 'pointerout'), c && n !== ea && (w = n.relatedTarget || n.fromElement) && (bn(w) || w[Ft]))) break e;
				if (
					(g || c) &&
					((c = f.window === f ? f : (c = f.ownerDocument) ? c.defaultView || c.parentWindow : window),
					g ? ((w = n.relatedTarget || n.toElement), (g = u), (w = w ? bn(w) : null), w !== null && ((S = Bn(w)), w !== S || (w.tag !== 5 && w.tag !== 6)) && (w = null)) : ((g = null), (w = u)),
					g !== w)
				) {
					if (
						((y = Vc),
						(E = 'onMouseLeave'),
						(m = 'onMouseEnter'),
						(h = 'mouse'),
						(e === 'pointerout' || e === 'pointerover') && ((y = Kc), (E = 'onPointerLeave'), (m = 'onPointerEnter'), (h = 'pointer')),
						(S = g == null ? c : or(g)),
						(x = w == null ? c : or(w)),
						(c = new y(E, h + 'leave', g, n, f)),
						(c.target = S),
						(c.relatedTarget = x),
						(E = null),
						bn(f) === u && ((y = new y(m, h + 'enter', w, n, f)), (y.target = x), (y.relatedTarget = S), (E = y)),
						(S = E),
						g && w)
					)
						t: {
							for (y = g, m = w, h = 0, x = y; x; x = Yn(x)) h++;
							for (x = 0, E = m; E; E = Yn(E)) x++;
							for (; 0 < h - x; ) (y = Yn(y)), h--;
							for (; 0 < x - h; ) (m = Yn(m)), x--;
							for (; h--; ) {
								if (y === m || (m !== null && y === m.alternate)) break t;
								(y = Yn(y)), (m = Yn(m));
							}
							y = null;
						}
					else y = null;
					g !== null && od(d, c, g, y, !1), w !== null && S !== null && od(d, S, w, y, !0);
				}
			}
			e: {
				if (((c = u ? or(u) : window), (g = c.nodeName && c.nodeName.toLowerCase()), g === 'select' || (g === 'input' && c.type === 'file'))) var C = c0;
				else if (Xc(c))
					if (yp) C = m0;
					else {
						C = f0;
						var T = d0;
					}
				else (g = c.nodeName) && g.toLowerCase() === 'input' && (c.type === 'checkbox' || c.type === 'radio') && (C = p0);
				if (C && (C = C(e, u))) {
					vp(d, C, n, f);
					break e;
				}
				T && T(e, c, u), e === 'focusout' && (T = c._wrapperState) && T.controlled && c.type === 'number' && Ql(c, 'number', c.value);
			}
			switch (((T = u ? or(u) : window), e)) {
				case 'focusin':
					(Xc(T) || T.contentEditable === 'true') && ((nr = T), (la = u), (lo = null));
					break;
				case 'focusout':
					lo = la = nr = null;
					break;
				case 'mousedown':
					aa = !0;
					break;
				case 'contextmenu':
				case 'mouseup':
				case 'dragend':
					(aa = !1), td(d, n, f);
					break;
				case 'selectionchange':
					if (v0) break;
				case 'keydown':
				case 'keyup':
					td(d, n, f);
			}
			var k;
			if (vu)
				e: {
					switch (e) {
						case 'compositionstart':
							var N = 'onCompositionStart';
							break e;
						case 'compositionend':
							N = 'onCompositionEnd';
							break e;
						case 'compositionupdate':
							N = 'onCompositionUpdate';
							break e;
					}
					N = void 0;
				}
			else tr ? hp(e, n) && (N = 'onCompositionEnd') : e === 'keydown' && n.keyCode === 229 && (N = 'onCompositionStart');
			N &&
				(mp && n.locale !== 'ko' && (tr || N !== 'onCompositionStart' ? N === 'onCompositionEnd' && tr && (k = pp()) : ((tn = f), (mu = 'value' in tn ? tn.value : tn.textContent), (tr = !0))),
				(T = Hi(u, N)),
				0 < T.length && ((N = new Gc(N, e, null, n, f)), d.push({event: N, listeners: T}), k ? (N.data = k) : ((k = gp(n)), k !== null && (N.data = k)))),
				(k = i0 ? s0(e, n) : l0(e, n)) && ((u = Hi(u, 'onBeforeInput')), 0 < u.length && ((f = new Gc('onBeforeInput', 'beforeinput', null, n, f)), d.push({event: f, listeners: u}), (f.data = k)));
		}
		Np(d, t);
	});
}
function Eo(e, t, n) {
	return {instance: e, listener: t, currentTarget: n};
}
function Hi(e, t) {
	for (var n = t + 'Capture', r = []; e !== null; ) {
		var o = e,
			i = o.stateNode;
		o.tag === 5 && i !== null && ((o = i), (i = ho(e, n)), i != null && r.unshift(Eo(e, i, o)), (i = ho(e, t)), i != null && r.push(Eo(e, i, o))), (e = e.return);
	}
	return r;
}
function Yn(e) {
	if (e === null) return null;
	do e = e.return;
	while (e && e.tag !== 5);
	return e || null;
}
function od(e, t, n, r, o) {
	for (var i = t._reactName, s = []; n !== null && n !== r; ) {
		var l = n,
			a = l.alternate,
			u = l.stateNode;
		if (a !== null && a === r) break;
		l.tag === 5 && u !== null && ((l = u), o ? ((a = ho(n, i)), a != null && s.unshift(Eo(n, a, l))) : o || ((a = ho(n, i)), a != null && s.push(Eo(n, a, l)))), (n = n.return);
	}
	s.length !== 0 && e.push({event: t, listeners: s});
}
var S0 = /\r\n?/g,
	E0 = /\u0000|\uFFFD/g;
function id(e) {
	return (typeof e == 'string' ? e : '' + e)
		.replace(
			S0,
			`
`
		)
		.replace(E0, '');
}
function si(e, t, n) {
	if (((t = id(t)), id(e) !== t && n)) throw Error(_(425));
}
function Wi() {}
var ua = null,
	ca = null;
function da(e, t) {
	return e === 'textarea' || e === 'noscript' || typeof t.children == 'string' || typeof t.children == 'number' || (typeof t.dangerouslySetInnerHTML == 'object' && t.dangerouslySetInnerHTML !== null && t.dangerouslySetInnerHTML.__html != null);
}
var fa = typeof setTimeout == 'function' ? setTimeout : void 0,
	C0 = typeof clearTimeout == 'function' ? clearTimeout : void 0,
	sd = typeof Promise == 'function' ? Promise : void 0,
	b0 =
		typeof queueMicrotask == 'function'
			? queueMicrotask
			: typeof sd < 'u'
			? function (e) {
					return sd.resolve(null).then(e).catch(k0);
			  }
			: fa;
function k0(e) {
	setTimeout(function () {
		throw e;
	});
}
function yl(e, t) {
	var n = t,
		r = 0;
	do {
		var o = n.nextSibling;
		if ((e.removeChild(n), o && o.nodeType === 8))
			if (((n = o.data), n === '/$')) {
				if (r === 0) {
					e.removeChild(o), yo(t);
					return;
				}
				r--;
			} else (n !== '$' && n !== '$?' && n !== '$!') || r++;
		n = o;
	} while (n);
	yo(t);
}
function ln(e) {
	for (; e != null; e = e.nextSibling) {
		var t = e.nodeType;
		if (t === 1 || t === 3) break;
		if (t === 8) {
			if (((t = e.data), t === '$' || t === '$!' || t === '$?')) break;
			if (t === '/$') return null;
		}
	}
	return e;
}
function ld(e) {
	e = e.previousSibling;
	for (var t = 0; e; ) {
		if (e.nodeType === 8) {
			var n = e.data;
			if (n === '$' || n === '$!' || n === '$?') {
				if (t === 0) return e;
				t--;
			} else n === '/$' && t++;
		}
		e = e.previousSibling;
	}
	return null;
}
var jr = Math.random().toString(36).slice(2),
	bt = '__reactFiber$' + jr,
	Co = '__reactProps$' + jr,
	Ft = '__reactContainer$' + jr,
	pa = '__reactEvents$' + jr,
	T0 = '__reactListeners$' + jr,
	P0 = '__reactHandles$' + jr;
function bn(e) {
	var t = e[bt];
	if (t) return t;
	for (var n = e.parentNode; n; ) {
		if ((t = n[Ft] || n[bt])) {
			if (((n = t.alternate), t.child !== null || (n !== null && n.child !== null)))
				for (e = ld(e); e !== null; ) {
					if ((n = e[bt])) return n;
					e = ld(e);
				}
			return t;
		}
		(e = n), (n = e.parentNode);
	}
	return null;
}
function $o(e) {
	return (e = e[bt] || e[Ft]), !e || (e.tag !== 5 && e.tag !== 6 && e.tag !== 13 && e.tag !== 3) ? null : e;
}
function or(e) {
	if (e.tag === 5 || e.tag === 6) return e.stateNode;
	throw Error(_(33));
}
function xs(e) {
	return e[Co] || null;
}
var ma = [],
	ir = -1;
function wn(e) {
	return {current: e};
}
function oe(e) {
	0 > ir || ((e.current = ma[ir]), (ma[ir] = null), ir--);
}
function ee(e, t) {
	ir++, (ma[ir] = e.current), (e.current = t);
}
var pn = {},
	Me = wn(pn),
	Be = wn(!1),
	An = pn;
function Er(e, t) {
	var n = e.type.contextTypes;
	if (!n) return pn;
	var r = e.stateNode;
	if (r && r.__reactInternalMemoizedUnmaskedChildContext === t) return r.__reactInternalMemoizedMaskedChildContext;
	var o = {},
		i;
	for (i in n) o[i] = t[i];
	return r && ((e = e.stateNode), (e.__reactInternalMemoizedUnmaskedChildContext = t), (e.__reactInternalMemoizedMaskedChildContext = o)), o;
}
function Ue(e) {
	return (e = e.childContextTypes), e != null;
}
function Vi() {
	oe(Be), oe(Me);
}
function ad(e, t, n) {
	if (Me.current !== pn) throw Error(_(168));
	ee(Me, t), ee(Be, n);
}
function Rp(e, t, n) {
	var r = e.stateNode;
	if (((t = t.childContextTypes), typeof r.getChildContext != 'function')) return n;
	r = r.getChildContext();
	for (var o in r) if (!(o in t)) throw Error(_(108, dy(e) || 'Unknown', o));
	return de({}, n, r);
}
function Gi(e) {
	return (e = ((e = e.stateNode) && e.__reactInternalMemoizedMergedChildContext) || pn), (An = Me.current), ee(Me, e), ee(Be, Be.current), !0;
}
function ud(e, t, n) {
	var r = e.stateNode;
	if (!r) throw Error(_(169));
	n ? ((e = Rp(e, t, An)), (r.__reactInternalMemoizedMergedChildContext = e), oe(Be), oe(Me), ee(Me, e)) : oe(Be), ee(Be, n);
}
var Ot = null,
	Ss = !1,
	wl = !1;
function Ap(e) {
	Ot === null ? (Ot = [e]) : Ot.push(e);
}
function N0(e) {
	(Ss = !0), Ap(e);
}
function xn() {
	if (!wl && Ot !== null) {
		wl = !0;
		var e = 0,
			t = q;
		try {
			var n = Ot;
			for (q = 1; e < n.length; e++) {
				var r = n[e];
				do r = r(!0);
				while (r !== null);
			}
			(Ot = null), (Ss = !1);
		} catch (o) {
			throw (Ot !== null && (Ot = Ot.slice(e + 1)), np(cu, xn), o);
		} finally {
			(q = t), (wl = !1);
		}
	}
	return null;
}
var sr = [],
	lr = 0,
	Ki = null,
	Yi = 0,
	Je = [],
	et = 0,
	Mn = null,
	jt = 1,
	It = '';
function En(e, t) {
	(sr[lr++] = Yi), (sr[lr++] = Ki), (Ki = e), (Yi = t);
}
function Mp(e, t, n) {
	(Je[et++] = jt), (Je[et++] = It), (Je[et++] = Mn), (Mn = e);
	var r = jt;
	e = It;
	var o = 32 - pt(r) - 1;
	(r &= ~(1 << o)), (n += 1);
	var i = 32 - pt(t) + o;
	if (30 < i) {
		var s = o - (o % 5);
		(i = (r & ((1 << s) - 1)).toString(32)), (r >>= s), (o -= s), (jt = (1 << (32 - pt(t) + o)) | (n << o) | r), (It = i + e);
	} else (jt = (1 << i) | (n << o) | r), (It = e);
}
function wu(e) {
	e.return !== null && (En(e, 1), Mp(e, 1, 0));
}
function xu(e) {
	for (; e === Ki; ) (Ki = sr[--lr]), (sr[lr] = null), (Yi = sr[--lr]), (sr[lr] = null);
	for (; e === Mn; ) (Mn = Je[--et]), (Je[et] = null), (It = Je[--et]), (Je[et] = null), (jt = Je[--et]), (Je[et] = null);
}
var Ye = null,
	Ke = null,
	ae = !1,
	dt = null;
function Op(e, t) {
	var n = tt(5, null, null, 0);
	(n.elementType = 'DELETED'), (n.stateNode = t), (n.return = e), (t = e.deletions), t === null ? ((e.deletions = [n]), (e.flags |= 16)) : t.push(n);
}
function cd(e, t) {
	switch (e.tag) {
		case 5:
			var n = e.type;
			return (t = t.nodeType !== 1 || n.toLowerCase() !== t.nodeName.toLowerCase() ? null : t), t !== null ? ((e.stateNode = t), (Ye = e), (Ke = ln(t.firstChild)), !0) : !1;
		case 6:
			return (t = e.pendingProps === '' || t.nodeType !== 3 ? null : t), t !== null ? ((e.stateNode = t), (Ye = e), (Ke = null), !0) : !1;
		case 13:
			return (
				(t = t.nodeType !== 8 ? null : t),
				t !== null
					? ((n = Mn !== null ? {id: jt, overflow: It} : null), (e.memoizedState = {dehydrated: t, treeContext: n, retryLane: 1073741824}), (n = tt(18, null, null, 0)), (n.stateNode = t), (n.return = e), (e.child = n), (Ye = e), (Ke = null), !0)
					: !1
			);
		default:
			return !1;
	}
}
function ha(e) {
	return (e.mode & 1) !== 0 && (e.flags & 128) === 0;
}
function ga(e) {
	if (ae) {
		var t = Ke;
		if (t) {
			var n = t;
			if (!cd(e, t)) {
				if (ha(e)) throw Error(_(418));
				t = ln(n.nextSibling);
				var r = Ye;
				t && cd(e, t) ? Op(r, n) : ((e.flags = (e.flags & -4097) | 2), (ae = !1), (Ye = e));
			}
		} else {
			if (ha(e)) throw Error(_(418));
			(e.flags = (e.flags & -4097) | 2), (ae = !1), (Ye = e);
		}
	}
}
function dd(e) {
	for (e = e.return; e !== null && e.tag !== 5 && e.tag !== 3 && e.tag !== 13; ) e = e.return;
	Ye = e;
}
function li(e) {
	if (e !== Ye) return !1;
	if (!ae) return dd(e), (ae = !0), !1;
	var t;
	if (((t = e.tag !== 3) && !(t = e.tag !== 5) && ((t = e.type), (t = t !== 'head' && t !== 'body' && !da(e.type, e.memoizedProps))), t && (t = Ke))) {
		if (ha(e)) throw (jp(), Error(_(418)));
		for (; t; ) Op(e, t), (t = ln(t.nextSibling));
	}
	if ((dd(e), e.tag === 13)) {
		if (((e = e.memoizedState), (e = e !== null ? e.dehydrated : null), !e)) throw Error(_(317));
		e: {
			for (e = e.nextSibling, t = 0; e; ) {
				if (e.nodeType === 8) {
					var n = e.data;
					if (n === '/$') {
						if (t === 0) {
							Ke = ln(e.nextSibling);
							break e;
						}
						t--;
					} else (n !== '$' && n !== '$!' && n !== '$?') || t++;
				}
				e = e.nextSibling;
			}
			Ke = null;
		}
	} else Ke = Ye ? ln(e.stateNode.nextSibling) : null;
	return !0;
}
function jp() {
	for (var e = Ke; e; ) e = ln(e.nextSibling);
}
function Cr() {
	(Ke = Ye = null), (ae = !1);
}
function Su(e) {
	dt === null ? (dt = [e]) : dt.push(e);
}
var _0 = Ht.ReactCurrentBatchConfig;
function Wr(e, t, n) {
	if (((e = n.ref), e !== null && typeof e != 'function' && typeof e != 'object')) {
		if (n._owner) {
			if (((n = n._owner), n)) {
				if (n.tag !== 1) throw Error(_(309));
				var r = n.stateNode;
			}
			if (!r) throw Error(_(147, e));
			var o = r,
				i = '' + e;
			return t !== null && t.ref !== null && typeof t.ref == 'function' && t.ref._stringRef === i
				? t.ref
				: ((t = function (s) {
						var l = o.refs;
						s === null ? delete l[i] : (l[i] = s);
				  }),
				  (t._stringRef = i),
				  t);
		}
		if (typeof e != 'string') throw Error(_(284));
		if (!n._owner) throw Error(_(290, e));
	}
	return e;
}
function ai(e, t) {
	throw ((e = Object.prototype.toString.call(t)), Error(_(31, e === '[object Object]' ? 'object with keys {' + Object.keys(t).join(', ') + '}' : e)));
}
function fd(e) {
	var t = e._init;
	return t(e._payload);
}
function Ip(e) {
	function t(m, h) {
		if (e) {
			var x = m.deletions;
			x === null ? ((m.deletions = [h]), (m.flags |= 16)) : x.push(h);
		}
	}
	function n(m, h) {
		if (!e) return null;
		for (; h !== null; ) t(m, h), (h = h.sibling);
		return null;
	}
	function r(m, h) {
		for (m = new Map(); h !== null; ) h.key !== null ? m.set(h.key, h) : m.set(h.index, h), (h = h.sibling);
		return m;
	}
	function o(m, h) {
		return (m = dn(m, h)), (m.index = 0), (m.sibling = null), m;
	}
	function i(m, h, x) {
		return (m.index = x), e ? ((x = m.alternate), x !== null ? ((x = x.index), x < h ? ((m.flags |= 2), h) : x) : ((m.flags |= 2), h)) : ((m.flags |= 1048576), h);
	}
	function s(m) {
		return e && m.alternate === null && (m.flags |= 2), m;
	}
	function l(m, h, x, E) {
		return h === null || h.tag !== 6 ? ((h = Tl(x, m.mode, E)), (h.return = m), h) : ((h = o(h, x)), (h.return = m), h);
	}
	function a(m, h, x, E) {
		var C = x.type;
		return C === er
			? f(m, h, x.props.children, E, x.key)
			: h !== null && (h.elementType === C || (typeof C == 'object' && C !== null && C.$$typeof === Zt && fd(C) === h.type))
			? ((E = o(h, x.props)), (E.ref = Wr(m, h, x)), (E.return = m), E)
			: ((E = Ai(x.type, x.key, x.props, null, m.mode, E)), (E.ref = Wr(m, h, x)), (E.return = m), E);
	}
	function u(m, h, x, E) {
		return h === null || h.tag !== 4 || h.stateNode.containerInfo !== x.containerInfo || h.stateNode.implementation !== x.implementation ? ((h = Pl(x, m.mode, E)), (h.return = m), h) : ((h = o(h, x.children || [])), (h.return = m), h);
	}
	function f(m, h, x, E, C) {
		return h === null || h.tag !== 7 ? ((h = Nn(x, m.mode, E, C)), (h.return = m), h) : ((h = o(h, x)), (h.return = m), h);
	}
	function d(m, h, x) {
		if ((typeof h == 'string' && h !== '') || typeof h == 'number') return (h = Tl('' + h, m.mode, x)), (h.return = m), h;
		if (typeof h == 'object' && h !== null) {
			switch (h.$$typeof) {
				case Zo:
					return (x = Ai(h.type, h.key, h.props, null, m.mode, x)), (x.ref = Wr(m, null, h)), (x.return = m), x;
				case Jn:
					return (h = Pl(h, m.mode, x)), (h.return = m), h;
				case Zt:
					var E = h._init;
					return d(m, E(h._payload), x);
			}
			if (Jr(h) || zr(h)) return (h = Nn(h, m.mode, x, null)), (h.return = m), h;
			ai(m, h);
		}
		return null;
	}
	function c(m, h, x, E) {
		var C = h !== null ? h.key : null;
		if ((typeof x == 'string' && x !== '') || typeof x == 'number') return C !== null ? null : l(m, h, '' + x, E);
		if (typeof x == 'object' && x !== null) {
			switch (x.$$typeof) {
				case Zo:
					return x.key === C ? a(m, h, x, E) : null;
				case Jn:
					return x.key === C ? u(m, h, x, E) : null;
				case Zt:
					return (C = x._init), c(m, h, C(x._payload), E);
			}
			if (Jr(x) || zr(x)) return C !== null ? null : f(m, h, x, E, null);
			ai(m, x);
		}
		return null;
	}
	function g(m, h, x, E, C) {
		if ((typeof E == 'string' && E !== '') || typeof E == 'number') return (m = m.get(x) || null), l(h, m, '' + E, C);
		if (typeof E == 'object' && E !== null) {
			switch (E.$$typeof) {
				case Zo:
					return (m = m.get(E.key === null ? x : E.key) || null), a(h, m, E, C);
				case Jn:
					return (m = m.get(E.key === null ? x : E.key) || null), u(h, m, E, C);
				case Zt:
					var T = E._init;
					return g(m, h, x, T(E._payload), C);
			}
			if (Jr(E) || zr(E)) return (m = m.get(x) || null), f(h, m, E, C, null);
			ai(h, E);
		}
		return null;
	}
	function w(m, h, x, E) {
		for (var C = null, T = null, k = h, N = (h = 0), D = null; k !== null && N < x.length; N++) {
			k.index > N ? ((D = k), (k = null)) : (D = k.sibling);
			var M = c(m, k, x[N], E);
			if (M === null) {
				k === null && (k = D);
				break;
			}
			e && k && M.alternate === null && t(m, k), (h = i(M, h, N)), T === null ? (C = M) : (T.sibling = M), (T = M), (k = D);
		}
		if (N === x.length) return n(m, k), ae && En(m, N), C;
		if (k === null) {
			for (; N < x.length; N++) (k = d(m, x[N], E)), k !== null && ((h = i(k, h, N)), T === null ? (C = k) : (T.sibling = k), (T = k));
			return ae && En(m, N), C;
		}
		for (k = r(m, k); N < x.length; N++) (D = g(k, m, N, x[N], E)), D !== null && (e && D.alternate !== null && k.delete(D.key === null ? N : D.key), (h = i(D, h, N)), T === null ? (C = D) : (T.sibling = D), (T = D));
		return (
			e &&
				k.forEach(function (z) {
					return t(m, z);
				}),
			ae && En(m, N),
			C
		);
	}
	function y(m, h, x, E) {
		var C = zr(x);
		if (typeof C != 'function') throw Error(_(150));
		if (((x = C.call(x)), x == null)) throw Error(_(151));
		for (var T = (C = null), k = h, N = (h = 0), D = null, M = x.next(); k !== null && !M.done; N++, M = x.next()) {
			k.index > N ? ((D = k), (k = null)) : (D = k.sibling);
			var z = c(m, k, M.value, E);
			if (z === null) {
				k === null && (k = D);
				break;
			}
			e && k && z.alternate === null && t(m, k), (h = i(z, h, N)), T === null ? (C = z) : (T.sibling = z), (T = z), (k = D);
		}
		if (M.done) return n(m, k), ae && En(m, N), C;
		if (k === null) {
			for (; !M.done; N++, M = x.next()) (M = d(m, M.value, E)), M !== null && ((h = i(M, h, N)), T === null ? (C = M) : (T.sibling = M), (T = M));
			return ae && En(m, N), C;
		}
		for (k = r(m, k); !M.done; N++, M = x.next()) (M = g(k, m, N, M.value, E)), M !== null && (e && M.alternate !== null && k.delete(M.key === null ? N : M.key), (h = i(M, h, N)), T === null ? (C = M) : (T.sibling = M), (T = M));
		return (
			e &&
				k.forEach(function (R) {
					return t(m, R);
				}),
			ae && En(m, N),
			C
		);
	}
	function S(m, h, x, E) {
		if ((typeof x == 'object' && x !== null && x.type === er && x.key === null && (x = x.props.children), typeof x == 'object' && x !== null)) {
			switch (x.$$typeof) {
				case Zo:
					e: {
						for (var C = x.key, T = h; T !== null; ) {
							if (T.key === C) {
								if (((C = x.type), C === er)) {
									if (T.tag === 7) {
										n(m, T.sibling), (h = o(T, x.props.children)), (h.return = m), (m = h);
										break e;
									}
								} else if (T.elementType === C || (typeof C == 'object' && C !== null && C.$$typeof === Zt && fd(C) === T.type)) {
									n(m, T.sibling), (h = o(T, x.props)), (h.ref = Wr(m, T, x)), (h.return = m), (m = h);
									break e;
								}
								n(m, T);
								break;
							} else t(m, T);
							T = T.sibling;
						}
						x.type === er ? ((h = Nn(x.props.children, m.mode, E, x.key)), (h.return = m), (m = h)) : ((E = Ai(x.type, x.key, x.props, null, m.mode, E)), (E.ref = Wr(m, h, x)), (E.return = m), (m = E));
					}
					return s(m);
				case Jn:
					e: {
						for (T = x.key; h !== null; ) {
							if (h.key === T)
								if (h.tag === 4 && h.stateNode.containerInfo === x.containerInfo && h.stateNode.implementation === x.implementation) {
									n(m, h.sibling), (h = o(h, x.children || [])), (h.return = m), (m = h);
									break e;
								} else {
									n(m, h);
									break;
								}
							else t(m, h);
							h = h.sibling;
						}
						(h = Pl(x, m.mode, E)), (h.return = m), (m = h);
					}
					return s(m);
				case Zt:
					return (T = x._init), S(m, h, T(x._payload), E);
			}
			if (Jr(x)) return w(m, h, x, E);
			if (zr(x)) return y(m, h, x, E);
			ai(m, x);
		}
		return (typeof x == 'string' && x !== '') || typeof x == 'number'
			? ((x = '' + x), h !== null && h.tag === 6 ? (n(m, h.sibling), (h = o(h, x)), (h.return = m), (m = h)) : (n(m, h), (h = Tl(x, m.mode, E)), (h.return = m), (m = h)), s(m))
			: n(m, h);
	}
	return S;
}
var br = Ip(!0),
	Dp = Ip(!1),
	Qi = wn(null),
	Xi = null,
	ar = null,
	Eu = null;
function Cu() {
	Eu = ar = Xi = null;
}
function bu(e) {
	var t = Qi.current;
	oe(Qi), (e._currentValue = t);
}
function va(e, t, n) {
	for (; e !== null; ) {
		var r = e.alternate;
		if (((e.childLanes & t) !== t ? ((e.childLanes |= t), r !== null && (r.childLanes |= t)) : r !== null && (r.childLanes & t) !== t && (r.childLanes |= t), e === n)) break;
		e = e.return;
	}
}
function hr(e, t) {
	(Xi = e), (Eu = ar = null), (e = e.dependencies), e !== null && e.firstContext !== null && (e.lanes & t && ($e = !0), (e.firstContext = null));
}
function ot(e) {
	var t = e._currentValue;
	if (Eu !== e)
		if (((e = {context: e, memoizedValue: t, next: null}), ar === null)) {
			if (Xi === null) throw Error(_(308));
			(ar = e), (Xi.dependencies = {lanes: 0, firstContext: e});
		} else ar = ar.next = e;
	return t;
}
var kn = null;
function ku(e) {
	kn === null ? (kn = [e]) : kn.push(e);
}
function Lp(e, t, n, r) {
	var o = t.interleaved;
	return o === null ? ((n.next = n), ku(t)) : ((n.next = o.next), (o.next = n)), (t.interleaved = n), zt(e, r);
}
function zt(e, t) {
	e.lanes |= t;
	var n = e.alternate;
	for (n !== null && (n.lanes |= t), n = e, e = e.return; e !== null; ) (e.childLanes |= t), (n = e.alternate), n !== null && (n.childLanes |= t), (n = e), (e = e.return);
	return n.tag === 3 ? n.stateNode : null;
}
var qt = !1;
function Tu(e) {
	e.updateQueue = {baseState: e.memoizedState, firstBaseUpdate: null, lastBaseUpdate: null, shared: {pending: null, interleaved: null, lanes: 0}, effects: null};
}
function Fp(e, t) {
	(e = e.updateQueue), t.updateQueue === e && (t.updateQueue = {baseState: e.baseState, firstBaseUpdate: e.firstBaseUpdate, lastBaseUpdate: e.lastBaseUpdate, shared: e.shared, effects: e.effects});
}
function Dt(e, t) {
	return {eventTime: e, lane: t, tag: 0, payload: null, callback: null, next: null};
}
function an(e, t, n) {
	var r = e.updateQueue;
	if (r === null) return null;
	if (((r = r.shared), Z & 2)) {
		var o = r.pending;
		return o === null ? (t.next = t) : ((t.next = o.next), (o.next = t)), (r.pending = t), zt(e, n);
	}
	return (o = r.interleaved), o === null ? ((t.next = t), ku(r)) : ((t.next = o.next), (o.next = t)), (r.interleaved = t), zt(e, n);
}
function ki(e, t, n) {
	if (((t = t.updateQueue), t !== null && ((t = t.shared), (n & 4194240) !== 0))) {
		var r = t.lanes;
		(r &= e.pendingLanes), (n |= r), (t.lanes = n), du(e, n);
	}
}
function pd(e, t) {
	var n = e.updateQueue,
		r = e.alternate;
	if (r !== null && ((r = r.updateQueue), n === r)) {
		var o = null,
			i = null;
		if (((n = n.firstBaseUpdate), n !== null)) {
			do {
				var s = {eventTime: n.eventTime, lane: n.lane, tag: n.tag, payload: n.payload, callback: n.callback, next: null};
				i === null ? (o = i = s) : (i = i.next = s), (n = n.next);
			} while (n !== null);
			i === null ? (o = i = t) : (i = i.next = t);
		} else o = i = t;
		(n = {baseState: r.baseState, firstBaseUpdate: o, lastBaseUpdate: i, shared: r.shared, effects: r.effects}), (e.updateQueue = n);
		return;
	}
	(e = n.lastBaseUpdate), e === null ? (n.firstBaseUpdate = t) : (e.next = t), (n.lastBaseUpdate = t);
}
function Zi(e, t, n, r) {
	var o = e.updateQueue;
	qt = !1;
	var i = o.firstBaseUpdate,
		s = o.lastBaseUpdate,
		l = o.shared.pending;
	if (l !== null) {
		o.shared.pending = null;
		var a = l,
			u = a.next;
		(a.next = null), s === null ? (i = u) : (s.next = u), (s = a);
		var f = e.alternate;
		f !== null && ((f = f.updateQueue), (l = f.lastBaseUpdate), l !== s && (l === null ? (f.firstBaseUpdate = u) : (l.next = u), (f.lastBaseUpdate = a)));
	}
	if (i !== null) {
		var d = o.baseState;
		(s = 0), (f = u = a = null), (l = i);
		do {
			var c = l.lane,
				g = l.eventTime;
			if ((r & c) === c) {
				f !== null && (f = f.next = {eventTime: g, lane: 0, tag: l.tag, payload: l.payload, callback: l.callback, next: null});
				e: {
					var w = e,
						y = l;
					switch (((c = t), (g = n), y.tag)) {
						case 1:
							if (((w = y.payload), typeof w == 'function')) {
								d = w.call(g, d, c);
								break e;
							}
							d = w;
							break e;
						case 3:
							w.flags = (w.flags & -65537) | 128;
						case 0:
							if (((w = y.payload), (c = typeof w == 'function' ? w.call(g, d, c) : w), c == null)) break e;
							d = de({}, d, c);
							break e;
						case 2:
							qt = !0;
					}
				}
				l.callback !== null && l.lane !== 0 && ((e.flags |= 64), (c = o.effects), c === null ? (o.effects = [l]) : c.push(l));
			} else (g = {eventTime: g, lane: c, tag: l.tag, payload: l.payload, callback: l.callback, next: null}), f === null ? ((u = f = g), (a = d)) : (f = f.next = g), (s |= c);
			if (((l = l.next), l === null)) {
				if (((l = o.shared.pending), l === null)) break;
				(c = l), (l = c.next), (c.next = null), (o.lastBaseUpdate = c), (o.shared.pending = null);
			}
		} while (!0);
		if ((f === null && (a = d), (o.baseState = a), (o.firstBaseUpdate = u), (o.lastBaseUpdate = f), (t = o.shared.interleaved), t !== null)) {
			o = t;
			do (s |= o.lane), (o = o.next);
			while (o !== t);
		} else i === null && (o.shared.lanes = 0);
		(jn |= s), (e.lanes = s), (e.memoizedState = d);
	}
}
function md(e, t, n) {
	if (((e = t.effects), (t.effects = null), e !== null))
		for (t = 0; t < e.length; t++) {
			var r = e[t],
				o = r.callback;
			if (o !== null) {
				if (((r.callback = null), (r = n), typeof o != 'function')) throw Error(_(191, o));
				o.call(r);
			}
		}
}
var Bo = {},
	Pt = wn(Bo),
	bo = wn(Bo),
	ko = wn(Bo);
function Tn(e) {
	if (e === Bo) throw Error(_(174));
	return e;
}
function Pu(e, t) {
	switch ((ee(ko, t), ee(bo, e), ee(Pt, Bo), (e = t.nodeType), e)) {
		case 9:
		case 11:
			t = (t = t.documentElement) ? t.namespaceURI : Zl(null, '');
			break;
		default:
			(e = e === 8 ? t.parentNode : t), (t = e.namespaceURI || null), (e = e.tagName), (t = Zl(t, e));
	}
	oe(Pt), ee(Pt, t);
}
function kr() {
	oe(Pt), oe(bo), oe(ko);
}
function zp(e) {
	Tn(ko.current);
	var t = Tn(Pt.current),
		n = Zl(t, e.type);
	t !== n && (ee(bo, e), ee(Pt, n));
}
function Nu(e) {
	bo.current === e && (oe(Pt), oe(bo));
}
var ue = wn(0);
function qi(e) {
	for (var t = e; t !== null; ) {
		if (t.tag === 13) {
			var n = t.memoizedState;
			if (n !== null && ((n = n.dehydrated), n === null || n.data === '$?' || n.data === '$!')) return t;
		} else if (t.tag === 19 && t.memoizedProps.revealOrder !== void 0) {
			if (t.flags & 128) return t;
		} else if (t.child !== null) {
			(t.child.return = t), (t = t.child);
			continue;
		}
		if (t === e) break;
		for (; t.sibling === null; ) {
			if (t.return === null || t.return === e) return null;
			t = t.return;
		}
		(t.sibling.return = t.return), (t = t.sibling);
	}
	return null;
}
var xl = [];
function _u() {
	for (var e = 0; e < xl.length; e++) xl[e]._workInProgressVersionPrimary = null;
	xl.length = 0;
}
var Ti = Ht.ReactCurrentDispatcher,
	Sl = Ht.ReactCurrentBatchConfig,
	On = 0,
	ce = null,
	ve = null,
	Ee = null,
	Ji = !1,
	ao = !1,
	To = 0,
	R0 = 0;
function _e() {
	throw Error(_(321));
}
function Ru(e, t) {
	if (t === null) return !1;
	for (var n = 0; n < t.length && n < e.length; n++) if (!ht(e[n], t[n])) return !1;
	return !0;
}
function Au(e, t, n, r, o, i) {
	if (((On = i), (ce = t), (t.memoizedState = null), (t.updateQueue = null), (t.lanes = 0), (Ti.current = e === null || e.memoizedState === null ? j0 : I0), (e = n(r, o)), ao)) {
		i = 0;
		do {
			if (((ao = !1), (To = 0), 25 <= i)) throw Error(_(301));
			(i += 1), (Ee = ve = null), (t.updateQueue = null), (Ti.current = D0), (e = n(r, o));
		} while (ao);
	}
	if (((Ti.current = es), (t = ve !== null && ve.next !== null), (On = 0), (Ee = ve = ce = null), (Ji = !1), t)) throw Error(_(300));
	return e;
}
function Mu() {
	var e = To !== 0;
	return (To = 0), e;
}
function Ct() {
	var e = {memoizedState: null, baseState: null, baseQueue: null, queue: null, next: null};
	return Ee === null ? (ce.memoizedState = Ee = e) : (Ee = Ee.next = e), Ee;
}
function it() {
	if (ve === null) {
		var e = ce.alternate;
		e = e !== null ? e.memoizedState : null;
	} else e = ve.next;
	var t = Ee === null ? ce.memoizedState : Ee.next;
	if (t !== null) (Ee = t), (ve = e);
	else {
		if (e === null) throw Error(_(310));
		(ve = e), (e = {memoizedState: ve.memoizedState, baseState: ve.baseState, baseQueue: ve.baseQueue, queue: ve.queue, next: null}), Ee === null ? (ce.memoizedState = Ee = e) : (Ee = Ee.next = e);
	}
	return Ee;
}
function Po(e, t) {
	return typeof t == 'function' ? t(e) : t;
}
function El(e) {
	var t = it(),
		n = t.queue;
	if (n === null) throw Error(_(311));
	n.lastRenderedReducer = e;
	var r = ve,
		o = r.baseQueue,
		i = n.pending;
	if (i !== null) {
		if (o !== null) {
			var s = o.next;
			(o.next = i.next), (i.next = s);
		}
		(r.baseQueue = o = i), (n.pending = null);
	}
	if (o !== null) {
		(i = o.next), (r = r.baseState);
		var l = (s = null),
			a = null,
			u = i;
		do {
			var f = u.lane;
			if ((On & f) === f) a !== null && (a = a.next = {lane: 0, action: u.action, hasEagerState: u.hasEagerState, eagerState: u.eagerState, next: null}), (r = u.hasEagerState ? u.eagerState : e(r, u.action));
			else {
				var d = {lane: f, action: u.action, hasEagerState: u.hasEagerState, eagerState: u.eagerState, next: null};
				a === null ? ((l = a = d), (s = r)) : (a = a.next = d), (ce.lanes |= f), (jn |= f);
			}
			u = u.next;
		} while (u !== null && u !== i);
		a === null ? (s = r) : (a.next = l), ht(r, t.memoizedState) || ($e = !0), (t.memoizedState = r), (t.baseState = s), (t.baseQueue = a), (n.lastRenderedState = r);
	}
	if (((e = n.interleaved), e !== null)) {
		o = e;
		do (i = o.lane), (ce.lanes |= i), (jn |= i), (o = o.next);
		while (o !== e);
	} else o === null && (n.lanes = 0);
	return [t.memoizedState, n.dispatch];
}
function Cl(e) {
	var t = it(),
		n = t.queue;
	if (n === null) throw Error(_(311));
	n.lastRenderedReducer = e;
	var r = n.dispatch,
		o = n.pending,
		i = t.memoizedState;
	if (o !== null) {
		n.pending = null;
		var s = (o = o.next);
		do (i = e(i, s.action)), (s = s.next);
		while (s !== o);
		ht(i, t.memoizedState) || ($e = !0), (t.memoizedState = i), t.baseQueue === null && (t.baseState = i), (n.lastRenderedState = i);
	}
	return [i, r];
}
function $p() {}
function Bp(e, t) {
	var n = ce,
		r = it(),
		o = t(),
		i = !ht(r.memoizedState, o);
	if ((i && ((r.memoizedState = o), ($e = !0)), (r = r.queue), Ou(Wp.bind(null, n, r, e), [e]), r.getSnapshot !== t || i || (Ee !== null && Ee.memoizedState.tag & 1))) {
		if (((n.flags |= 2048), No(9, Hp.bind(null, n, r, o, t), void 0, null), Ce === null)) throw Error(_(349));
		On & 30 || Up(n, t, o);
	}
	return o;
}
function Up(e, t, n) {
	(e.flags |= 16384), (e = {getSnapshot: t, value: n}), (t = ce.updateQueue), t === null ? ((t = {lastEffect: null, stores: null}), (ce.updateQueue = t), (t.stores = [e])) : ((n = t.stores), n === null ? (t.stores = [e]) : n.push(e));
}
function Hp(e, t, n, r) {
	(t.value = n), (t.getSnapshot = r), Vp(t) && Gp(e);
}
function Wp(e, t, n) {
	return n(function () {
		Vp(t) && Gp(e);
	});
}
function Vp(e) {
	var t = e.getSnapshot;
	e = e.value;
	try {
		var n = t();
		return !ht(e, n);
	} catch {
		return !0;
	}
}
function Gp(e) {
	var t = zt(e, 1);
	t !== null && mt(t, e, 1, -1);
}
function hd(e) {
	var t = Ct();
	return (
		typeof e == 'function' && (e = e()),
		(t.memoizedState = t.baseState = e),
		(e = {pending: null, interleaved: null, lanes: 0, dispatch: null, lastRenderedReducer: Po, lastRenderedState: e}),
		(t.queue = e),
		(e = e.dispatch = O0.bind(null, ce, e)),
		[t.memoizedState, e]
	);
}
function No(e, t, n, r) {
	return (
		(e = {tag: e, create: t, destroy: n, deps: r, next: null}),
		(t = ce.updateQueue),
		t === null ? ((t = {lastEffect: null, stores: null}), (ce.updateQueue = t), (t.lastEffect = e.next = e)) : ((n = t.lastEffect), n === null ? (t.lastEffect = e.next = e) : ((r = n.next), (n.next = e), (e.next = r), (t.lastEffect = e))),
		e
	);
}
function Kp() {
	return it().memoizedState;
}
function Pi(e, t, n, r) {
	var o = Ct();
	(ce.flags |= e), (o.memoizedState = No(1 | t, n, void 0, r === void 0 ? null : r));
}
function Es(e, t, n, r) {
	var o = it();
	r = r === void 0 ? null : r;
	var i = void 0;
	if (ve !== null) {
		var s = ve.memoizedState;
		if (((i = s.destroy), r !== null && Ru(r, s.deps))) {
			o.memoizedState = No(t, n, i, r);
			return;
		}
	}
	(ce.flags |= e), (o.memoizedState = No(1 | t, n, i, r));
}
function gd(e, t) {
	return Pi(8390656, 8, e, t);
}
function Ou(e, t) {
	return Es(2048, 8, e, t);
}
function Yp(e, t) {
	return Es(4, 2, e, t);
}
function Qp(e, t) {
	return Es(4, 4, e, t);
}
function Xp(e, t) {
	if (typeof t == 'function')
		return (
			(e = e()),
			t(e),
			function () {
				t(null);
			}
		);
	if (t != null)
		return (
			(e = e()),
			(t.current = e),
			function () {
				t.current = null;
			}
		);
}
function Zp(e, t, n) {
	return (n = n != null ? n.concat([e]) : null), Es(4, 4, Xp.bind(null, t, e), n);
}
function ju() {}
function qp(e, t) {
	var n = it();
	t = t === void 0 ? null : t;
	var r = n.memoizedState;
	return r !== null && t !== null && Ru(t, r[1]) ? r[0] : ((n.memoizedState = [e, t]), e);
}
function Jp(e, t) {
	var n = it();
	t = t === void 0 ? null : t;
	var r = n.memoizedState;
	return r !== null && t !== null && Ru(t, r[1]) ? r[0] : ((e = e()), (n.memoizedState = [e, t]), e);
}
function em(e, t, n) {
	return On & 21 ? (ht(n, t) || ((n = ip()), (ce.lanes |= n), (jn |= n), (e.baseState = !0)), t) : (e.baseState && ((e.baseState = !1), ($e = !0)), (e.memoizedState = n));
}
function A0(e, t) {
	var n = q;
	(q = n !== 0 && 4 > n ? n : 4), e(!0);
	var r = Sl.transition;
	Sl.transition = {};
	try {
		e(!1), t();
	} finally {
		(q = n), (Sl.transition = r);
	}
}
function tm() {
	return it().memoizedState;
}
function M0(e, t, n) {
	var r = cn(e);
	if (((n = {lane: r, action: n, hasEagerState: !1, eagerState: null, next: null}), nm(e))) rm(t, n);
	else if (((n = Lp(e, t, n, r)), n !== null)) {
		var o = je();
		mt(n, e, r, o), om(n, t, r);
	}
}
function O0(e, t, n) {
	var r = cn(e),
		o = {lane: r, action: n, hasEagerState: !1, eagerState: null, next: null};
	if (nm(e)) rm(t, o);
	else {
		var i = e.alternate;
		if (e.lanes === 0 && (i === null || i.lanes === 0) && ((i = t.lastRenderedReducer), i !== null))
			try {
				var s = t.lastRenderedState,
					l = i(s, n);
				if (((o.hasEagerState = !0), (o.eagerState = l), ht(l, s))) {
					var a = t.interleaved;
					a === null ? ((o.next = o), ku(t)) : ((o.next = a.next), (a.next = o)), (t.interleaved = o);
					return;
				}
			} catch {
			} finally {
			}
		(n = Lp(e, t, o, r)), n !== null && ((o = je()), mt(n, e, r, o), om(n, t, r));
	}
}
function nm(e) {
	var t = e.alternate;
	return e === ce || (t !== null && t === ce);
}
function rm(e, t) {
	ao = Ji = !0;
	var n = e.pending;
	n === null ? (t.next = t) : ((t.next = n.next), (n.next = t)), (e.pending = t);
}
function om(e, t, n) {
	if (n & 4194240) {
		var r = t.lanes;
		(r &= e.pendingLanes), (n |= r), (t.lanes = n), du(e, n);
	}
}
var es = {
		readContext: ot,
		useCallback: _e,
		useContext: _e,
		useEffect: _e,
		useImperativeHandle: _e,
		useInsertionEffect: _e,
		useLayoutEffect: _e,
		useMemo: _e,
		useReducer: _e,
		useRef: _e,
		useState: _e,
		useDebugValue: _e,
		useDeferredValue: _e,
		useTransition: _e,
		useMutableSource: _e,
		useSyncExternalStore: _e,
		useId: _e,
		unstable_isNewReconciler: !1,
	},
	j0 = {
		readContext: ot,
		useCallback: function (e, t) {
			return (Ct().memoizedState = [e, t === void 0 ? null : t]), e;
		},
		useContext: ot,
		useEffect: gd,
		useImperativeHandle: function (e, t, n) {
			return (n = n != null ? n.concat([e]) : null), Pi(4194308, 4, Xp.bind(null, t, e), n);
		},
		useLayoutEffect: function (e, t) {
			return Pi(4194308, 4, e, t);
		},
		useInsertionEffect: function (e, t) {
			return Pi(4, 2, e, t);
		},
		useMemo: function (e, t) {
			var n = Ct();
			return (t = t === void 0 ? null : t), (e = e()), (n.memoizedState = [e, t]), e;
		},
		useReducer: function (e, t, n) {
			var r = Ct();
			return (
				(t = n !== void 0 ? n(t) : t),
				(r.memoizedState = r.baseState = t),
				(e = {pending: null, interleaved: null, lanes: 0, dispatch: null, lastRenderedReducer: e, lastRenderedState: t}),
				(r.queue = e),
				(e = e.dispatch = M0.bind(null, ce, e)),
				[r.memoizedState, e]
			);
		},
		useRef: function (e) {
			var t = Ct();
			return (e = {current: e}), (t.memoizedState = e);
		},
		useState: hd,
		useDebugValue: ju,
		useDeferredValue: function (e) {
			return (Ct().memoizedState = e);
		},
		useTransition: function () {
			var e = hd(!1),
				t = e[0];
			return (e = A0.bind(null, e[1])), (Ct().memoizedState = e), [t, e];
		},
		useMutableSource: function () {},
		useSyncExternalStore: function (e, t, n) {
			var r = ce,
				o = Ct();
			if (ae) {
				if (n === void 0) throw Error(_(407));
				n = n();
			} else {
				if (((n = t()), Ce === null)) throw Error(_(349));
				On & 30 || Up(r, t, n);
			}
			o.memoizedState = n;
			var i = {value: n, getSnapshot: t};
			return (o.queue = i), gd(Wp.bind(null, r, i, e), [e]), (r.flags |= 2048), No(9, Hp.bind(null, r, i, n, t), void 0, null), n;
		},
		useId: function () {
			var e = Ct(),
				t = Ce.identifierPrefix;
			if (ae) {
				var n = It,
					r = jt;
				(n = (r & ~(1 << (32 - pt(r) - 1))).toString(32) + n), (t = ':' + t + 'R' + n), (n = To++), 0 < n && (t += 'H' + n.toString(32)), (t += ':');
			} else (n = R0++), (t = ':' + t + 'r' + n.toString(32) + ':');
			return (e.memoizedState = t);
		},
		unstable_isNewReconciler: !1,
	},
	I0 = {
		readContext: ot,
		useCallback: qp,
		useContext: ot,
		useEffect: Ou,
		useImperativeHandle: Zp,
		useInsertionEffect: Yp,
		useLayoutEffect: Qp,
		useMemo: Jp,
		useReducer: El,
		useRef: Kp,
		useState: function () {
			return El(Po);
		},
		useDebugValue: ju,
		useDeferredValue: function (e) {
			var t = it();
			return em(t, ve.memoizedState, e);
		},
		useTransition: function () {
			var e = El(Po)[0],
				t = it().memoizedState;
			return [e, t];
		},
		useMutableSource: $p,
		useSyncExternalStore: Bp,
		useId: tm,
		unstable_isNewReconciler: !1,
	},
	D0 = {
		readContext: ot,
		useCallback: qp,
		useContext: ot,
		useEffect: Ou,
		useImperativeHandle: Zp,
		useInsertionEffect: Yp,
		useLayoutEffect: Qp,
		useMemo: Jp,
		useReducer: Cl,
		useRef: Kp,
		useState: function () {
			return Cl(Po);
		},
		useDebugValue: ju,
		useDeferredValue: function (e) {
			var t = it();
			return ve === null ? (t.memoizedState = e) : em(t, ve.memoizedState, e);
		},
		useTransition: function () {
			var e = Cl(Po)[0],
				t = it().memoizedState;
			return [e, t];
		},
		useMutableSource: $p,
		useSyncExternalStore: Bp,
		useId: tm,
		unstable_isNewReconciler: !1,
	};
function ut(e, t) {
	if (e && e.defaultProps) {
		(t = de({}, t)), (e = e.defaultProps);
		for (var n in e) t[n] === void 0 && (t[n] = e[n]);
		return t;
	}
	return t;
}
function ya(e, t, n, r) {
	(t = e.memoizedState), (n = n(r, t)), (n = n == null ? t : de({}, t, n)), (e.memoizedState = n), e.lanes === 0 && (e.updateQueue.baseState = n);
}
var Cs = {
	isMounted: function (e) {
		return (e = e._reactInternals) ? Bn(e) === e : !1;
	},
	enqueueSetState: function (e, t, n) {
		e = e._reactInternals;
		var r = je(),
			o = cn(e),
			i = Dt(r, o);
		(i.payload = t), n != null && (i.callback = n), (t = an(e, i, o)), t !== null && (mt(t, e, o, r), ki(t, e, o));
	},
	enqueueReplaceState: function (e, t, n) {
		e = e._reactInternals;
		var r = je(),
			o = cn(e),
			i = Dt(r, o);
		(i.tag = 1), (i.payload = t), n != null && (i.callback = n), (t = an(e, i, o)), t !== null && (mt(t, e, o, r), ki(t, e, o));
	},
	enqueueForceUpdate: function (e, t) {
		e = e._reactInternals;
		var n = je(),
			r = cn(e),
			o = Dt(n, r);
		(o.tag = 2), t != null && (o.callback = t), (t = an(e, o, r)), t !== null && (mt(t, e, r, n), ki(t, e, r));
	},
};
function vd(e, t, n, r, o, i, s) {
	return (e = e.stateNode), typeof e.shouldComponentUpdate == 'function' ? e.shouldComponentUpdate(r, i, s) : t.prototype && t.prototype.isPureReactComponent ? !xo(n, r) || !xo(o, i) : !0;
}
function im(e, t, n) {
	var r = !1,
		o = pn,
		i = t.contextType;
	return (
		typeof i == 'object' && i !== null ? (i = ot(i)) : ((o = Ue(t) ? An : Me.current), (r = t.contextTypes), (i = (r = r != null) ? Er(e, o) : pn)),
		(t = new t(n, i)),
		(e.memoizedState = t.state !== null && t.state !== void 0 ? t.state : null),
		(t.updater = Cs),
		(e.stateNode = t),
		(t._reactInternals = e),
		r && ((e = e.stateNode), (e.__reactInternalMemoizedUnmaskedChildContext = o), (e.__reactInternalMemoizedMaskedChildContext = i)),
		t
	);
}
function yd(e, t, n, r) {
	(e = t.state),
		typeof t.componentWillReceiveProps == 'function' && t.componentWillReceiveProps(n, r),
		typeof t.UNSAFE_componentWillReceiveProps == 'function' && t.UNSAFE_componentWillReceiveProps(n, r),
		t.state !== e && Cs.enqueueReplaceState(t, t.state, null);
}
function wa(e, t, n, r) {
	var o = e.stateNode;
	(o.props = n), (o.state = e.memoizedState), (o.refs = {}), Tu(e);
	var i = t.contextType;
	typeof i == 'object' && i !== null ? (o.context = ot(i)) : ((i = Ue(t) ? An : Me.current), (o.context = Er(e, i))),
		(o.state = e.memoizedState),
		(i = t.getDerivedStateFromProps),
		typeof i == 'function' && (ya(e, t, i, n), (o.state = e.memoizedState)),
		typeof t.getDerivedStateFromProps == 'function' ||
			typeof o.getSnapshotBeforeUpdate == 'function' ||
			(typeof o.UNSAFE_componentWillMount != 'function' && typeof o.componentWillMount != 'function') ||
			((t = o.state),
			typeof o.componentWillMount == 'function' && o.componentWillMount(),
			typeof o.UNSAFE_componentWillMount == 'function' && o.UNSAFE_componentWillMount(),
			t !== o.state && Cs.enqueueReplaceState(o, o.state, null),
			Zi(e, n, o, r),
			(o.state = e.memoizedState)),
		typeof o.componentDidMount == 'function' && (e.flags |= 4194308);
}
function Tr(e, t) {
	try {
		var n = '',
			r = t;
		do (n += cy(r)), (r = r.return);
		while (r);
		var o = n;
	} catch (i) {
		o =
			`
Error generating stack: ` +
			i.message +
			`
` +
			i.stack;
	}
	return {value: e, source: t, stack: o, digest: null};
}
function bl(e, t, n) {
	return {value: e, source: null, stack: n ?? null, digest: t ?? null};
}
function xa(e, t) {
	try {
		console.error(t.value);
	} catch (n) {
		setTimeout(function () {
			throw n;
		});
	}
}
var L0 = typeof WeakMap == 'function' ? WeakMap : Map;
function sm(e, t, n) {
	(n = Dt(-1, n)), (n.tag = 3), (n.payload = {element: null});
	var r = t.value;
	return (
		(n.callback = function () {
			ns || ((ns = !0), (Ra = r)), xa(e, t);
		}),
		n
	);
}
function lm(e, t, n) {
	(n = Dt(-1, n)), (n.tag = 3);
	var r = e.type.getDerivedStateFromError;
	if (typeof r == 'function') {
		var o = t.value;
		(n.payload = function () {
			return r(o);
		}),
			(n.callback = function () {
				xa(e, t);
			});
	}
	var i = e.stateNode;
	return (
		i !== null &&
			typeof i.componentDidCatch == 'function' &&
			(n.callback = function () {
				xa(e, t), typeof r != 'function' && (un === null ? (un = new Set([this])) : un.add(this));
				var s = t.stack;
				this.componentDidCatch(t.value, {componentStack: s !== null ? s : ''});
			}),
		n
	);
}
function wd(e, t, n) {
	var r = e.pingCache;
	if (r === null) {
		r = e.pingCache = new L0();
		var o = new Set();
		r.set(t, o);
	} else (o = r.get(t)), o === void 0 && ((o = new Set()), r.set(t, o));
	o.has(n) || (o.add(n), (e = Z0.bind(null, e, t, n)), t.then(e, e));
}
function xd(e) {
	do {
		var t;
		if (((t = e.tag === 13) && ((t = e.memoizedState), (t = t !== null ? t.dehydrated !== null : !0)), t)) return e;
		e = e.return;
	} while (e !== null);
	return null;
}
function Sd(e, t, n, r, o) {
	return e.mode & 1
		? ((e.flags |= 65536), (e.lanes = o), e)
		: (e === t ? (e.flags |= 65536) : ((e.flags |= 128), (n.flags |= 131072), (n.flags &= -52805), n.tag === 1 && (n.alternate === null ? (n.tag = 17) : ((t = Dt(-1, 1)), (t.tag = 2), an(n, t, 1))), (n.lanes |= 1)), e);
}
var F0 = Ht.ReactCurrentOwner,
	$e = !1;
function Oe(e, t, n, r) {
	t.child = e === null ? Dp(t, null, n, r) : br(t, e.child, n, r);
}
function Ed(e, t, n, r, o) {
	n = n.render;
	var i = t.ref;
	return hr(t, o), (r = Au(e, t, n, r, i, o)), (n = Mu()), e !== null && !$e ? ((t.updateQueue = e.updateQueue), (t.flags &= -2053), (e.lanes &= ~o), $t(e, t, o)) : (ae && n && wu(t), (t.flags |= 1), Oe(e, t, r, o), t.child);
}
function Cd(e, t, n, r, o) {
	if (e === null) {
		var i = n.type;
		return typeof i == 'function' && !Uu(i) && i.defaultProps === void 0 && n.compare === null && n.defaultProps === void 0
			? ((t.tag = 15), (t.type = i), am(e, t, i, r, o))
			: ((e = Ai(n.type, null, r, t, t.mode, o)), (e.ref = t.ref), (e.return = t), (t.child = e));
	}
	if (((i = e.child), !(e.lanes & o))) {
		var s = i.memoizedProps;
		if (((n = n.compare), (n = n !== null ? n : xo), n(s, r) && e.ref === t.ref)) return $t(e, t, o);
	}
	return (t.flags |= 1), (e = dn(i, r)), (e.ref = t.ref), (e.return = t), (t.child = e);
}
function am(e, t, n, r, o) {
	if (e !== null) {
		var i = e.memoizedProps;
		if (xo(i, r) && e.ref === t.ref)
			if ((($e = !1), (t.pendingProps = r = i), (e.lanes & o) !== 0)) e.flags & 131072 && ($e = !0);
			else return (t.lanes = e.lanes), $t(e, t, o);
	}
	return Sa(e, t, n, r, o);
}
function um(e, t, n) {
	var r = t.pendingProps,
		o = r.children,
		i = e !== null ? e.memoizedState : null;
	if (r.mode === 'hidden')
		if (!(t.mode & 1)) (t.memoizedState = {baseLanes: 0, cachePool: null, transitions: null}), ee(cr, Ve), (Ve |= n);
		else {
			if (!(n & 1073741824)) return (e = i !== null ? i.baseLanes | n : n), (t.lanes = t.childLanes = 1073741824), (t.memoizedState = {baseLanes: e, cachePool: null, transitions: null}), (t.updateQueue = null), ee(cr, Ve), (Ve |= e), null;
			(t.memoizedState = {baseLanes: 0, cachePool: null, transitions: null}), (r = i !== null ? i.baseLanes : n), ee(cr, Ve), (Ve |= r);
		}
	else i !== null ? ((r = i.baseLanes | n), (t.memoizedState = null)) : (r = n), ee(cr, Ve), (Ve |= r);
	return Oe(e, t, o, n), t.child;
}
function cm(e, t) {
	var n = t.ref;
	((e === null && n !== null) || (e !== null && e.ref !== n)) && ((t.flags |= 512), (t.flags |= 2097152));
}
function Sa(e, t, n, r, o) {
	var i = Ue(n) ? An : Me.current;
	return (i = Er(t, i)), hr(t, o), (n = Au(e, t, n, r, i, o)), (r = Mu()), e !== null && !$e ? ((t.updateQueue = e.updateQueue), (t.flags &= -2053), (e.lanes &= ~o), $t(e, t, o)) : (ae && r && wu(t), (t.flags |= 1), Oe(e, t, n, o), t.child);
}
function bd(e, t, n, r, o) {
	if (Ue(n)) {
		var i = !0;
		Gi(t);
	} else i = !1;
	if ((hr(t, o), t.stateNode === null)) Ni(e, t), im(t, n, r), wa(t, n, r, o), (r = !0);
	else if (e === null) {
		var s = t.stateNode,
			l = t.memoizedProps;
		s.props = l;
		var a = s.context,
			u = n.contextType;
		typeof u == 'object' && u !== null ? (u = ot(u)) : ((u = Ue(n) ? An : Me.current), (u = Er(t, u)));
		var f = n.getDerivedStateFromProps,
			d = typeof f == 'function' || typeof s.getSnapshotBeforeUpdate == 'function';
		d || (typeof s.UNSAFE_componentWillReceiveProps != 'function' && typeof s.componentWillReceiveProps != 'function') || ((l !== r || a !== u) && yd(t, s, r, u)), (qt = !1);
		var c = t.memoizedState;
		(s.state = c),
			Zi(t, r, s, o),
			(a = t.memoizedState),
			l !== r || c !== a || Be.current || qt
				? (typeof f == 'function' && (ya(t, n, f, r), (a = t.memoizedState)),
				  (l = qt || vd(t, n, l, r, c, a, u))
						? (d ||
								(typeof s.UNSAFE_componentWillMount != 'function' && typeof s.componentWillMount != 'function') ||
								(typeof s.componentWillMount == 'function' && s.componentWillMount(), typeof s.UNSAFE_componentWillMount == 'function' && s.UNSAFE_componentWillMount()),
						  typeof s.componentDidMount == 'function' && (t.flags |= 4194308))
						: (typeof s.componentDidMount == 'function' && (t.flags |= 4194308), (t.memoizedProps = r), (t.memoizedState = a)),
				  (s.props = r),
				  (s.state = a),
				  (s.context = u),
				  (r = l))
				: (typeof s.componentDidMount == 'function' && (t.flags |= 4194308), (r = !1));
	} else {
		(s = t.stateNode),
			Fp(e, t),
			(l = t.memoizedProps),
			(u = t.type === t.elementType ? l : ut(t.type, l)),
			(s.props = u),
			(d = t.pendingProps),
			(c = s.context),
			(a = n.contextType),
			typeof a == 'object' && a !== null ? (a = ot(a)) : ((a = Ue(n) ? An : Me.current), (a = Er(t, a)));
		var g = n.getDerivedStateFromProps;
		(f = typeof g == 'function' || typeof s.getSnapshotBeforeUpdate == 'function') || (typeof s.UNSAFE_componentWillReceiveProps != 'function' && typeof s.componentWillReceiveProps != 'function') || ((l !== d || c !== a) && yd(t, s, r, a)),
			(qt = !1),
			(c = t.memoizedState),
			(s.state = c),
			Zi(t, r, s, o);
		var w = t.memoizedState;
		l !== d || c !== w || Be.current || qt
			? (typeof g == 'function' && (ya(t, n, g, r), (w = t.memoizedState)),
			  (u = qt || vd(t, n, u, r, c, w, a) || !1)
					? (f ||
							(typeof s.UNSAFE_componentWillUpdate != 'function' && typeof s.componentWillUpdate != 'function') ||
							(typeof s.componentWillUpdate == 'function' && s.componentWillUpdate(r, w, a), typeof s.UNSAFE_componentWillUpdate == 'function' && s.UNSAFE_componentWillUpdate(r, w, a)),
					  typeof s.componentDidUpdate == 'function' && (t.flags |= 4),
					  typeof s.getSnapshotBeforeUpdate == 'function' && (t.flags |= 1024))
					: (typeof s.componentDidUpdate != 'function' || (l === e.memoizedProps && c === e.memoizedState) || (t.flags |= 4),
					  typeof s.getSnapshotBeforeUpdate != 'function' || (l === e.memoizedProps && c === e.memoizedState) || (t.flags |= 1024),
					  (t.memoizedProps = r),
					  (t.memoizedState = w)),
			  (s.props = r),
			  (s.state = w),
			  (s.context = a),
			  (r = u))
			: (typeof s.componentDidUpdate != 'function' || (l === e.memoizedProps && c === e.memoizedState) || (t.flags |= 4),
			  typeof s.getSnapshotBeforeUpdate != 'function' || (l === e.memoizedProps && c === e.memoizedState) || (t.flags |= 1024),
			  (r = !1));
	}
	return Ea(e, t, n, r, i, o);
}
function Ea(e, t, n, r, o, i) {
	cm(e, t);
	var s = (t.flags & 128) !== 0;
	if (!r && !s) return o && ud(t, n, !1), $t(e, t, i);
	(r = t.stateNode), (F0.current = t);
	var l = s && typeof n.getDerivedStateFromError != 'function' ? null : r.render();
	return (t.flags |= 1), e !== null && s ? ((t.child = br(t, e.child, null, i)), (t.child = br(t, null, l, i))) : Oe(e, t, l, i), (t.memoizedState = r.state), o && ud(t, n, !0), t.child;
}
function dm(e) {
	var t = e.stateNode;
	t.pendingContext ? ad(e, t.pendingContext, t.pendingContext !== t.context) : t.context && ad(e, t.context, !1), Pu(e, t.containerInfo);
}
function kd(e, t, n, r, o) {
	return Cr(), Su(o), (t.flags |= 256), Oe(e, t, n, r), t.child;
}
var Ca = {dehydrated: null, treeContext: null, retryLane: 0};
function ba(e) {
	return {baseLanes: e, cachePool: null, transitions: null};
}
function fm(e, t, n) {
	var r = t.pendingProps,
		o = ue.current,
		i = !1,
		s = (t.flags & 128) !== 0,
		l;
	if (((l = s) || (l = e !== null && e.memoizedState === null ? !1 : (o & 2) !== 0), l ? ((i = !0), (t.flags &= -129)) : (e === null || e.memoizedState !== null) && (o |= 1), ee(ue, o & 1), e === null))
		return (
			ga(t),
			(e = t.memoizedState),
			e !== null && ((e = e.dehydrated), e !== null)
				? (t.mode & 1 ? (e.data === '$!' ? (t.lanes = 8) : (t.lanes = 1073741824)) : (t.lanes = 1), null)
				: ((s = r.children),
				  (e = r.fallback),
				  i
						? ((r = t.mode),
						  (i = t.child),
						  (s = {mode: 'hidden', children: s}),
						  !(r & 1) && i !== null ? ((i.childLanes = 0), (i.pendingProps = s)) : (i = Ts(s, r, 0, null)),
						  (e = Nn(e, r, n, null)),
						  (i.return = t),
						  (e.return = t),
						  (i.sibling = e),
						  (t.child = i),
						  (t.child.memoizedState = ba(n)),
						  (t.memoizedState = Ca),
						  e)
						: Iu(t, s))
		);
	if (((o = e.memoizedState), o !== null && ((l = o.dehydrated), l !== null))) return z0(e, t, s, r, l, o, n);
	if (i) {
		(i = r.fallback), (s = t.mode), (o = e.child), (l = o.sibling);
		var a = {mode: 'hidden', children: r.children};
		return (
			!(s & 1) && t.child !== o ? ((r = t.child), (r.childLanes = 0), (r.pendingProps = a), (t.deletions = null)) : ((r = dn(o, a)), (r.subtreeFlags = o.subtreeFlags & 14680064)),
			l !== null ? (i = dn(l, i)) : ((i = Nn(i, s, n, null)), (i.flags |= 2)),
			(i.return = t),
			(r.return = t),
			(r.sibling = i),
			(t.child = r),
			(r = i),
			(i = t.child),
			(s = e.child.memoizedState),
			(s = s === null ? ba(n) : {baseLanes: s.baseLanes | n, cachePool: null, transitions: s.transitions}),
			(i.memoizedState = s),
			(i.childLanes = e.childLanes & ~n),
			(t.memoizedState = Ca),
			r
		);
	}
	return (
		(i = e.child),
		(e = i.sibling),
		(r = dn(i, {mode: 'visible', children: r.children})),
		!(t.mode & 1) && (r.lanes = n),
		(r.return = t),
		(r.sibling = null),
		e !== null && ((n = t.deletions), n === null ? ((t.deletions = [e]), (t.flags |= 16)) : n.push(e)),
		(t.child = r),
		(t.memoizedState = null),
		r
	);
}
function Iu(e, t) {
	return (t = Ts({mode: 'visible', children: t}, e.mode, 0, null)), (t.return = e), (e.child = t);
}
function ui(e, t, n, r) {
	return r !== null && Su(r), br(t, e.child, null, n), (e = Iu(t, t.pendingProps.children)), (e.flags |= 2), (t.memoizedState = null), e;
}
function z0(e, t, n, r, o, i, s) {
	if (n)
		return t.flags & 256
			? ((t.flags &= -257), (r = bl(Error(_(422)))), ui(e, t, s, r))
			: t.memoizedState !== null
			? ((t.child = e.child), (t.flags |= 128), null)
			: ((i = r.fallback),
			  (o = t.mode),
			  (r = Ts({mode: 'visible', children: r.children}, o, 0, null)),
			  (i = Nn(i, o, s, null)),
			  (i.flags |= 2),
			  (r.return = t),
			  (i.return = t),
			  (r.sibling = i),
			  (t.child = r),
			  t.mode & 1 && br(t, e.child, null, s),
			  (t.child.memoizedState = ba(s)),
			  (t.memoizedState = Ca),
			  i);
	if (!(t.mode & 1)) return ui(e, t, s, null);
	if (o.data === '$!') {
		if (((r = o.nextSibling && o.nextSibling.dataset), r)) var l = r.dgst;
		return (r = l), (i = Error(_(419))), (r = bl(i, r, void 0)), ui(e, t, s, r);
	}
	if (((l = (s & e.childLanes) !== 0), $e || l)) {
		if (((r = Ce), r !== null)) {
			switch (s & -s) {
				case 4:
					o = 2;
					break;
				case 16:
					o = 8;
					break;
				case 64:
				case 128:
				case 256:
				case 512:
				case 1024:
				case 2048:
				case 4096:
				case 8192:
				case 16384:
				case 32768:
				case 65536:
				case 131072:
				case 262144:
				case 524288:
				case 1048576:
				case 2097152:
				case 4194304:
				case 8388608:
				case 16777216:
				case 33554432:
				case 67108864:
					o = 32;
					break;
				case 536870912:
					o = 268435456;
					break;
				default:
					o = 0;
			}
			(o = o & (r.suspendedLanes | s) ? 0 : o), o !== 0 && o !== i.retryLane && ((i.retryLane = o), zt(e, o), mt(r, e, o, -1));
		}
		return Bu(), (r = bl(Error(_(421)))), ui(e, t, s, r);
	}
	return o.data === '$?'
		? ((t.flags |= 128), (t.child = e.child), (t = q0.bind(null, e)), (o._reactRetry = t), null)
		: ((e = i.treeContext), (Ke = ln(o.nextSibling)), (Ye = t), (ae = !0), (dt = null), e !== null && ((Je[et++] = jt), (Je[et++] = It), (Je[et++] = Mn), (jt = e.id), (It = e.overflow), (Mn = t)), (t = Iu(t, r.children)), (t.flags |= 4096), t);
}
function Td(e, t, n) {
	e.lanes |= t;
	var r = e.alternate;
	r !== null && (r.lanes |= t), va(e.return, t, n);
}
function kl(e, t, n, r, o) {
	var i = e.memoizedState;
	i === null ? (e.memoizedState = {isBackwards: t, rendering: null, renderingStartTime: 0, last: r, tail: n, tailMode: o}) : ((i.isBackwards = t), (i.rendering = null), (i.renderingStartTime = 0), (i.last = r), (i.tail = n), (i.tailMode = o));
}
function pm(e, t, n) {
	var r = t.pendingProps,
		o = r.revealOrder,
		i = r.tail;
	if ((Oe(e, t, r.children, n), (r = ue.current), r & 2)) (r = (r & 1) | 2), (t.flags |= 128);
	else {
		if (e !== null && e.flags & 128)
			e: for (e = t.child; e !== null; ) {
				if (e.tag === 13) e.memoizedState !== null && Td(e, n, t);
				else if (e.tag === 19) Td(e, n, t);
				else if (e.child !== null) {
					(e.child.return = e), (e = e.child);
					continue;
				}
				if (e === t) break e;
				for (; e.sibling === null; ) {
					if (e.return === null || e.return === t) break e;
					e = e.return;
				}
				(e.sibling.return = e.return), (e = e.sibling);
			}
		r &= 1;
	}
	if ((ee(ue, r), !(t.mode & 1))) t.memoizedState = null;
	else
		switch (o) {
			case 'forwards':
				for (n = t.child, o = null; n !== null; ) (e = n.alternate), e !== null && qi(e) === null && (o = n), (n = n.sibling);
				(n = o), n === null ? ((o = t.child), (t.child = null)) : ((o = n.sibling), (n.sibling = null)), kl(t, !1, o, n, i);
				break;
			case 'backwards':
				for (n = null, o = t.child, t.child = null; o !== null; ) {
					if (((e = o.alternate), e !== null && qi(e) === null)) {
						t.child = o;
						break;
					}
					(e = o.sibling), (o.sibling = n), (n = o), (o = e);
				}
				kl(t, !0, n, null, i);
				break;
			case 'together':
				kl(t, !1, null, null, void 0);
				break;
			default:
				t.memoizedState = null;
		}
	return t.child;
}
function Ni(e, t) {
	!(t.mode & 1) && e !== null && ((e.alternate = null), (t.alternate = null), (t.flags |= 2));
}
function $t(e, t, n) {
	if ((e !== null && (t.dependencies = e.dependencies), (jn |= t.lanes), !(n & t.childLanes))) return null;
	if (e !== null && t.child !== e.child) throw Error(_(153));
	if (t.child !== null) {
		for (e = t.child, n = dn(e, e.pendingProps), t.child = n, n.return = t; e.sibling !== null; ) (e = e.sibling), (n = n.sibling = dn(e, e.pendingProps)), (n.return = t);
		n.sibling = null;
	}
	return t.child;
}
function $0(e, t, n) {
	switch (t.tag) {
		case 3:
			dm(t), Cr();
			break;
		case 5:
			zp(t);
			break;
		case 1:
			Ue(t.type) && Gi(t);
			break;
		case 4:
			Pu(t, t.stateNode.containerInfo);
			break;
		case 10:
			var r = t.type._context,
				o = t.memoizedProps.value;
			ee(Qi, r._currentValue), (r._currentValue = o);
			break;
		case 13:
			if (((r = t.memoizedState), r !== null)) return r.dehydrated !== null ? (ee(ue, ue.current & 1), (t.flags |= 128), null) : n & t.child.childLanes ? fm(e, t, n) : (ee(ue, ue.current & 1), (e = $t(e, t, n)), e !== null ? e.sibling : null);
			ee(ue, ue.current & 1);
			break;
		case 19:
			if (((r = (n & t.childLanes) !== 0), e.flags & 128)) {
				if (r) return pm(e, t, n);
				t.flags |= 128;
			}
			if (((o = t.memoizedState), o !== null && ((o.rendering = null), (o.tail = null), (o.lastEffect = null)), ee(ue, ue.current), r)) break;
			return null;
		case 22:
		case 23:
			return (t.lanes = 0), um(e, t, n);
	}
	return $t(e, t, n);
}
var mm, ka, hm, gm;
mm = function (e, t) {
	for (var n = t.child; n !== null; ) {
		if (n.tag === 5 || n.tag === 6) e.appendChild(n.stateNode);
		else if (n.tag !== 4 && n.child !== null) {
			(n.child.return = n), (n = n.child);
			continue;
		}
		if (n === t) break;
		for (; n.sibling === null; ) {
			if (n.return === null || n.return === t) return;
			n = n.return;
		}
		(n.sibling.return = n.return), (n = n.sibling);
	}
};
ka = function () {};
hm = function (e, t, n, r) {
	var o = e.memoizedProps;
	if (o !== r) {
		(e = t.stateNode), Tn(Pt.current);
		var i = null;
		switch (n) {
			case 'input':
				(o = Kl(e, o)), (r = Kl(e, r)), (i = []);
				break;
			case 'select':
				(o = de({}, o, {value: void 0})), (r = de({}, r, {value: void 0})), (i = []);
				break;
			case 'textarea':
				(o = Xl(e, o)), (r = Xl(e, r)), (i = []);
				break;
			default:
				typeof o.onClick != 'function' && typeof r.onClick == 'function' && (e.onclick = Wi);
		}
		ql(n, r);
		var s;
		n = null;
		for (u in o)
			if (!r.hasOwnProperty(u) && o.hasOwnProperty(u) && o[u] != null)
				if (u === 'style') {
					var l = o[u];
					for (s in l) l.hasOwnProperty(s) && (n || (n = {}), (n[s] = ''));
				} else u !== 'dangerouslySetInnerHTML' && u !== 'children' && u !== 'suppressContentEditableWarning' && u !== 'suppressHydrationWarning' && u !== 'autoFocus' && (po.hasOwnProperty(u) ? i || (i = []) : (i = i || []).push(u, null));
		for (u in r) {
			var a = r[u];
			if (((l = o != null ? o[u] : void 0), r.hasOwnProperty(u) && a !== l && (a != null || l != null)))
				if (u === 'style')
					if (l) {
						for (s in l) !l.hasOwnProperty(s) || (a && a.hasOwnProperty(s)) || (n || (n = {}), (n[s] = ''));
						for (s in a) a.hasOwnProperty(s) && l[s] !== a[s] && (n || (n = {}), (n[s] = a[s]));
					} else n || (i || (i = []), i.push(u, n)), (n = a);
				else
					u === 'dangerouslySetInnerHTML'
						? ((a = a ? a.__html : void 0), (l = l ? l.__html : void 0), a != null && l !== a && (i = i || []).push(u, a))
						: u === 'children'
						? (typeof a != 'string' && typeof a != 'number') || (i = i || []).push(u, '' + a)
						: u !== 'suppressContentEditableWarning' && u !== 'suppressHydrationWarning' && (po.hasOwnProperty(u) ? (a != null && u === 'onScroll' && re('scroll', e), i || l === a || (i = [])) : (i = i || []).push(u, a));
		}
		n && (i = i || []).push('style', n);
		var u = i;
		(t.updateQueue = u) && (t.flags |= 4);
	}
};
gm = function (e, t, n, r) {
	n !== r && (t.flags |= 4);
};
function Vr(e, t) {
	if (!ae)
		switch (e.tailMode) {
			case 'hidden':
				t = e.tail;
				for (var n = null; t !== null; ) t.alternate !== null && (n = t), (t = t.sibling);
				n === null ? (e.tail = null) : (n.sibling = null);
				break;
			case 'collapsed':
				n = e.tail;
				for (var r = null; n !== null; ) n.alternate !== null && (r = n), (n = n.sibling);
				r === null ? (t || e.tail === null ? (e.tail = null) : (e.tail.sibling = null)) : (r.sibling = null);
		}
}
function Re(e) {
	var t = e.alternate !== null && e.alternate.child === e.child,
		n = 0,
		r = 0;
	if (t) for (var o = e.child; o !== null; ) (n |= o.lanes | o.childLanes), (r |= o.subtreeFlags & 14680064), (r |= o.flags & 14680064), (o.return = e), (o = o.sibling);
	else for (o = e.child; o !== null; ) (n |= o.lanes | o.childLanes), (r |= o.subtreeFlags), (r |= o.flags), (o.return = e), (o = o.sibling);
	return (e.subtreeFlags |= r), (e.childLanes = n), t;
}
function B0(e, t, n) {
	var r = t.pendingProps;
	switch ((xu(t), t.tag)) {
		case 2:
		case 16:
		case 15:
		case 0:
		case 11:
		case 7:
		case 8:
		case 12:
		case 9:
		case 14:
			return Re(t), null;
		case 1:
			return Ue(t.type) && Vi(), Re(t), null;
		case 3:
			return (
				(r = t.stateNode),
				kr(),
				oe(Be),
				oe(Me),
				_u(),
				r.pendingContext && ((r.context = r.pendingContext), (r.pendingContext = null)),
				(e === null || e.child === null) && (li(t) ? (t.flags |= 4) : e === null || (e.memoizedState.isDehydrated && !(t.flags & 256)) || ((t.flags |= 1024), dt !== null && (Oa(dt), (dt = null)))),
				ka(e, t),
				Re(t),
				null
			);
		case 5:
			Nu(t);
			var o = Tn(ko.current);
			if (((n = t.type), e !== null && t.stateNode != null)) hm(e, t, n, r, o), e.ref !== t.ref && ((t.flags |= 512), (t.flags |= 2097152));
			else {
				if (!r) {
					if (t.stateNode === null) throw Error(_(166));
					return Re(t), null;
				}
				if (((e = Tn(Pt.current)), li(t))) {
					(r = t.stateNode), (n = t.type);
					var i = t.memoizedProps;
					switch (((r[bt] = t), (r[Co] = i), (e = (t.mode & 1) !== 0), n)) {
						case 'dialog':
							re('cancel', r), re('close', r);
							break;
						case 'iframe':
						case 'object':
						case 'embed':
							re('load', r);
							break;
						case 'video':
						case 'audio':
							for (o = 0; o < to.length; o++) re(to[o], r);
							break;
						case 'source':
							re('error', r);
							break;
						case 'img':
						case 'image':
						case 'link':
							re('error', r), re('load', r);
							break;
						case 'details':
							re('toggle', r);
							break;
						case 'input':
							Ic(r, i), re('invalid', r);
							break;
						case 'select':
							(r._wrapperState = {wasMultiple: !!i.multiple}), re('invalid', r);
							break;
						case 'textarea':
							Lc(r, i), re('invalid', r);
					}
					ql(n, i), (o = null);
					for (var s in i)
						if (i.hasOwnProperty(s)) {
							var l = i[s];
							s === 'children'
								? typeof l == 'string'
									? r.textContent !== l && (i.suppressHydrationWarning !== !0 && si(r.textContent, l, e), (o = ['children', l]))
									: typeof l == 'number' && r.textContent !== '' + l && (i.suppressHydrationWarning !== !0 && si(r.textContent, l, e), (o = ['children', '' + l]))
								: po.hasOwnProperty(s) && l != null && s === 'onScroll' && re('scroll', r);
						}
					switch (n) {
						case 'input':
							qo(r), Dc(r, i, !0);
							break;
						case 'textarea':
							qo(r), Fc(r);
							break;
						case 'select':
						case 'option':
							break;
						default:
							typeof i.onClick == 'function' && (r.onclick = Wi);
					}
					(r = o), (t.updateQueue = r), r !== null && (t.flags |= 4);
				} else {
					(s = o.nodeType === 9 ? o : o.ownerDocument),
						e === 'http://www.w3.org/1999/xhtml' && (e = Wf(n)),
						e === 'http://www.w3.org/1999/xhtml'
							? n === 'script'
								? ((e = s.createElement('div')), (e.innerHTML = '<script></script>'), (e = e.removeChild(e.firstChild)))
								: typeof r.is == 'string'
								? (e = s.createElement(n, {is: r.is}))
								: ((e = s.createElement(n)), n === 'select' && ((s = e), r.multiple ? (s.multiple = !0) : r.size && (s.size = r.size)))
							: (e = s.createElementNS(e, n)),
						(e[bt] = t),
						(e[Co] = r),
						mm(e, t, !1, !1),
						(t.stateNode = e);
					e: {
						switch (((s = Jl(n, r)), n)) {
							case 'dialog':
								re('cancel', e), re('close', e), (o = r);
								break;
							case 'iframe':
							case 'object':
							case 'embed':
								re('load', e), (o = r);
								break;
							case 'video':
							case 'audio':
								for (o = 0; o < to.length; o++) re(to[o], e);
								o = r;
								break;
							case 'source':
								re('error', e), (o = r);
								break;
							case 'img':
							case 'image':
							case 'link':
								re('error', e), re('load', e), (o = r);
								break;
							case 'details':
								re('toggle', e), (o = r);
								break;
							case 'input':
								Ic(e, r), (o = Kl(e, r)), re('invalid', e);
								break;
							case 'option':
								o = r;
								break;
							case 'select':
								(e._wrapperState = {wasMultiple: !!r.multiple}), (o = de({}, r, {value: void 0})), re('invalid', e);
								break;
							case 'textarea':
								Lc(e, r), (o = Xl(e, r)), re('invalid', e);
								break;
							default:
								o = r;
						}
						ql(n, o), (l = o);
						for (i in l)
							if (l.hasOwnProperty(i)) {
								var a = l[i];
								i === 'style'
									? Kf(e, a)
									: i === 'dangerouslySetInnerHTML'
									? ((a = a ? a.__html : void 0), a != null && Vf(e, a))
									: i === 'children'
									? typeof a == 'string'
										? (n !== 'textarea' || a !== '') && mo(e, a)
										: typeof a == 'number' && mo(e, '' + a)
									: i !== 'suppressContentEditableWarning' && i !== 'suppressHydrationWarning' && i !== 'autoFocus' && (po.hasOwnProperty(i) ? a != null && i === 'onScroll' && re('scroll', e) : a != null && iu(e, i, a, s));
							}
						switch (n) {
							case 'input':
								qo(e), Dc(e, r, !1);
								break;
							case 'textarea':
								qo(e), Fc(e);
								break;
							case 'option':
								r.value != null && e.setAttribute('value', '' + fn(r.value));
								break;
							case 'select':
								(e.multiple = !!r.multiple), (i = r.value), i != null ? dr(e, !!r.multiple, i, !1) : r.defaultValue != null && dr(e, !!r.multiple, r.defaultValue, !0);
								break;
							default:
								typeof o.onClick == 'function' && (e.onclick = Wi);
						}
						switch (n) {
							case 'button':
							case 'input':
							case 'select':
							case 'textarea':
								r = !!r.autoFocus;
								break e;
							case 'img':
								r = !0;
								break e;
							default:
								r = !1;
						}
					}
					r && (t.flags |= 4);
				}
				t.ref !== null && ((t.flags |= 512), (t.flags |= 2097152));
			}
			return Re(t), null;
		case 6:
			if (e && t.stateNode != null) gm(e, t, e.memoizedProps, r);
			else {
				if (typeof r != 'string' && t.stateNode === null) throw Error(_(166));
				if (((n = Tn(ko.current)), Tn(Pt.current), li(t))) {
					if (((r = t.stateNode), (n = t.memoizedProps), (r[bt] = t), (i = r.nodeValue !== n) && ((e = Ye), e !== null)))
						switch (e.tag) {
							case 3:
								si(r.nodeValue, n, (e.mode & 1) !== 0);
								break;
							case 5:
								e.memoizedProps.suppressHydrationWarning !== !0 && si(r.nodeValue, n, (e.mode & 1) !== 0);
						}
					i && (t.flags |= 4);
				} else (r = (n.nodeType === 9 ? n : n.ownerDocument).createTextNode(r)), (r[bt] = t), (t.stateNode = r);
			}
			return Re(t), null;
		case 13:
			if ((oe(ue), (r = t.memoizedState), e === null || (e.memoizedState !== null && e.memoizedState.dehydrated !== null))) {
				if (ae && Ke !== null && t.mode & 1 && !(t.flags & 128)) jp(), Cr(), (t.flags |= 98560), (i = !1);
				else if (((i = li(t)), r !== null && r.dehydrated !== null)) {
					if (e === null) {
						if (!i) throw Error(_(318));
						if (((i = t.memoizedState), (i = i !== null ? i.dehydrated : null), !i)) throw Error(_(317));
						i[bt] = t;
					} else Cr(), !(t.flags & 128) && (t.memoizedState = null), (t.flags |= 4);
					Re(t), (i = !1);
				} else dt !== null && (Oa(dt), (dt = null)), (i = !0);
				if (!i) return t.flags & 65536 ? t : null;
			}
			return t.flags & 128
				? ((t.lanes = n), t)
				: ((r = r !== null), r !== (e !== null && e.memoizedState !== null) && r && ((t.child.flags |= 8192), t.mode & 1 && (e === null || ue.current & 1 ? ye === 0 && (ye = 3) : Bu())), t.updateQueue !== null && (t.flags |= 4), Re(t), null);
		case 4:
			return kr(), ka(e, t), e === null && So(t.stateNode.containerInfo), Re(t), null;
		case 10:
			return bu(t.type._context), Re(t), null;
		case 17:
			return Ue(t.type) && Vi(), Re(t), null;
		case 19:
			if ((oe(ue), (i = t.memoizedState), i === null)) return Re(t), null;
			if (((r = (t.flags & 128) !== 0), (s = i.rendering), s === null))
				if (r) Vr(i, !1);
				else {
					if (ye !== 0 || (e !== null && e.flags & 128))
						for (e = t.child; e !== null; ) {
							if (((s = qi(e)), s !== null)) {
								for (t.flags |= 128, Vr(i, !1), r = s.updateQueue, r !== null && ((t.updateQueue = r), (t.flags |= 4)), t.subtreeFlags = 0, r = n, n = t.child; n !== null; )
									(i = n),
										(e = r),
										(i.flags &= 14680066),
										(s = i.alternate),
										s === null
											? ((i.childLanes = 0), (i.lanes = e), (i.child = null), (i.subtreeFlags = 0), (i.memoizedProps = null), (i.memoizedState = null), (i.updateQueue = null), (i.dependencies = null), (i.stateNode = null))
											: ((i.childLanes = s.childLanes),
											  (i.lanes = s.lanes),
											  (i.child = s.child),
											  (i.subtreeFlags = 0),
											  (i.deletions = null),
											  (i.memoizedProps = s.memoizedProps),
											  (i.memoizedState = s.memoizedState),
											  (i.updateQueue = s.updateQueue),
											  (i.type = s.type),
											  (e = s.dependencies),
											  (i.dependencies = e === null ? null : {lanes: e.lanes, firstContext: e.firstContext})),
										(n = n.sibling);
								return ee(ue, (ue.current & 1) | 2), t.child;
							}
							e = e.sibling;
						}
					i.tail !== null && me() > Pr && ((t.flags |= 128), (r = !0), Vr(i, !1), (t.lanes = 4194304));
				}
			else {
				if (!r)
					if (((e = qi(s)), e !== null)) {
						if (((t.flags |= 128), (r = !0), (n = e.updateQueue), n !== null && ((t.updateQueue = n), (t.flags |= 4)), Vr(i, !0), i.tail === null && i.tailMode === 'hidden' && !s.alternate && !ae)) return Re(t), null;
					} else 2 * me() - i.renderingStartTime > Pr && n !== 1073741824 && ((t.flags |= 128), (r = !0), Vr(i, !1), (t.lanes = 4194304));
				i.isBackwards ? ((s.sibling = t.child), (t.child = s)) : ((n = i.last), n !== null ? (n.sibling = s) : (t.child = s), (i.last = s));
			}
			return i.tail !== null ? ((t = i.tail), (i.rendering = t), (i.tail = t.sibling), (i.renderingStartTime = me()), (t.sibling = null), (n = ue.current), ee(ue, r ? (n & 1) | 2 : n & 1), t) : (Re(t), null);
		case 22:
		case 23:
			return $u(), (r = t.memoizedState !== null), e !== null && (e.memoizedState !== null) !== r && (t.flags |= 8192), r && t.mode & 1 ? Ve & 1073741824 && (Re(t), t.subtreeFlags & 6 && (t.flags |= 8192)) : Re(t), null;
		case 24:
			return null;
		case 25:
			return null;
	}
	throw Error(_(156, t.tag));
}
function U0(e, t) {
	switch ((xu(t), t.tag)) {
		case 1:
			return Ue(t.type) && Vi(), (e = t.flags), e & 65536 ? ((t.flags = (e & -65537) | 128), t) : null;
		case 3:
			return kr(), oe(Be), oe(Me), _u(), (e = t.flags), e & 65536 && !(e & 128) ? ((t.flags = (e & -65537) | 128), t) : null;
		case 5:
			return Nu(t), null;
		case 13:
			if ((oe(ue), (e = t.memoizedState), e !== null && e.dehydrated !== null)) {
				if (t.alternate === null) throw Error(_(340));
				Cr();
			}
			return (e = t.flags), e & 65536 ? ((t.flags = (e & -65537) | 128), t) : null;
		case 19:
			return oe(ue), null;
		case 4:
			return kr(), null;
		case 10:
			return bu(t.type._context), null;
		case 22:
		case 23:
			return $u(), null;
		case 24:
			return null;
		default:
			return null;
	}
}
var ci = !1,
	Ae = !1,
	H0 = typeof WeakSet == 'function' ? WeakSet : Set,
	I = null;
function ur(e, t) {
	var n = e.ref;
	if (n !== null)
		if (typeof n == 'function')
			try {
				n(null);
			} catch (r) {
				fe(e, t, r);
			}
		else n.current = null;
}
function Ta(e, t, n) {
	try {
		n();
	} catch (r) {
		fe(e, t, r);
	}
}
var Pd = !1;
function W0(e, t) {
	if (((ua = Bi), (e = Sp()), yu(e))) {
		if ('selectionStart' in e) var n = {start: e.selectionStart, end: e.selectionEnd};
		else
			e: {
				n = ((n = e.ownerDocument) && n.defaultView) || window;
				var r = n.getSelection && n.getSelection();
				if (r && r.rangeCount !== 0) {
					n = r.anchorNode;
					var o = r.anchorOffset,
						i = r.focusNode;
					r = r.focusOffset;
					try {
						n.nodeType, i.nodeType;
					} catch {
						n = null;
						break e;
					}
					var s = 0,
						l = -1,
						a = -1,
						u = 0,
						f = 0,
						d = e,
						c = null;
					t: for (;;) {
						for (var g; d !== n || (o !== 0 && d.nodeType !== 3) || (l = s + o), d !== i || (r !== 0 && d.nodeType !== 3) || (a = s + r), d.nodeType === 3 && (s += d.nodeValue.length), (g = d.firstChild) !== null; ) (c = d), (d = g);
						for (;;) {
							if (d === e) break t;
							if ((c === n && ++u === o && (l = s), c === i && ++f === r && (a = s), (g = d.nextSibling) !== null)) break;
							(d = c), (c = d.parentNode);
						}
						d = g;
					}
					n = l === -1 || a === -1 ? null : {start: l, end: a};
				} else n = null;
			}
		n = n || {start: 0, end: 0};
	} else n = null;
	for (ca = {focusedElem: e, selectionRange: n}, Bi = !1, I = t; I !== null; )
		if (((t = I), (e = t.child), (t.subtreeFlags & 1028) !== 0 && e !== null)) (e.return = t), (I = e);
		else
			for (; I !== null; ) {
				t = I;
				try {
					var w = t.alternate;
					if (t.flags & 1024)
						switch (t.tag) {
							case 0:
							case 11:
							case 15:
								break;
							case 1:
								if (w !== null) {
									var y = w.memoizedProps,
										S = w.memoizedState,
										m = t.stateNode,
										h = m.getSnapshotBeforeUpdate(t.elementType === t.type ? y : ut(t.type, y), S);
									m.__reactInternalSnapshotBeforeUpdate = h;
								}
								break;
							case 3:
								var x = t.stateNode.containerInfo;
								x.nodeType === 1 ? (x.textContent = '') : x.nodeType === 9 && x.documentElement && x.removeChild(x.documentElement);
								break;
							case 5:
							case 6:
							case 4:
							case 17:
								break;
							default:
								throw Error(_(163));
						}
				} catch (E) {
					fe(t, t.return, E);
				}
				if (((e = t.sibling), e !== null)) {
					(e.return = t.return), (I = e);
					break;
				}
				I = t.return;
			}
	return (w = Pd), (Pd = !1), w;
}
function uo(e, t, n) {
	var r = t.updateQueue;
	if (((r = r !== null ? r.lastEffect : null), r !== null)) {
		var o = (r = r.next);
		do {
			if ((o.tag & e) === e) {
				var i = o.destroy;
				(o.destroy = void 0), i !== void 0 && Ta(t, n, i);
			}
			o = o.next;
		} while (o !== r);
	}
}
function bs(e, t) {
	if (((t = t.updateQueue), (t = t !== null ? t.lastEffect : null), t !== null)) {
		var n = (t = t.next);
		do {
			if ((n.tag & e) === e) {
				var r = n.create;
				n.destroy = r();
			}
			n = n.next;
		} while (n !== t);
	}
}
function Pa(e) {
	var t = e.ref;
	if (t !== null) {
		var n = e.stateNode;
		switch (e.tag) {
			case 5:
				e = n;
				break;
			default:
				e = n;
		}
		typeof t == 'function' ? t(e) : (t.current = e);
	}
}
function vm(e) {
	var t = e.alternate;
	t !== null && ((e.alternate = null), vm(t)),
		(e.child = null),
		(e.deletions = null),
		(e.sibling = null),
		e.tag === 5 && ((t = e.stateNode), t !== null && (delete t[bt], delete t[Co], delete t[pa], delete t[T0], delete t[P0])),
		(e.stateNode = null),
		(e.return = null),
		(e.dependencies = null),
		(e.memoizedProps = null),
		(e.memoizedState = null),
		(e.pendingProps = null),
		(e.stateNode = null),
		(e.updateQueue = null);
}
function ym(e) {
	return e.tag === 5 || e.tag === 3 || e.tag === 4;
}
function Nd(e) {
	e: for (;;) {
		for (; e.sibling === null; ) {
			if (e.return === null || ym(e.return)) return null;
			e = e.return;
		}
		for (e.sibling.return = e.return, e = e.sibling; e.tag !== 5 && e.tag !== 6 && e.tag !== 18; ) {
			if (e.flags & 2 || e.child === null || e.tag === 4) continue e;
			(e.child.return = e), (e = e.child);
		}
		if (!(e.flags & 2)) return e.stateNode;
	}
}
function Na(e, t, n) {
	var r = e.tag;
	if (r === 5 || r === 6)
		(e = e.stateNode),
			t
				? n.nodeType === 8
					? n.parentNode.insertBefore(e, t)
					: n.insertBefore(e, t)
				: (n.nodeType === 8 ? ((t = n.parentNode), t.insertBefore(e, n)) : ((t = n), t.appendChild(e)), (n = n._reactRootContainer), n != null || t.onclick !== null || (t.onclick = Wi));
	else if (r !== 4 && ((e = e.child), e !== null)) for (Na(e, t, n), e = e.sibling; e !== null; ) Na(e, t, n), (e = e.sibling);
}
function _a(e, t, n) {
	var r = e.tag;
	if (r === 5 || r === 6) (e = e.stateNode), t ? n.insertBefore(e, t) : n.appendChild(e);
	else if (r !== 4 && ((e = e.child), e !== null)) for (_a(e, t, n), e = e.sibling; e !== null; ) _a(e, t, n), (e = e.sibling);
}
var ke = null,
	ct = !1;
function Kt(e, t, n) {
	for (n = n.child; n !== null; ) wm(e, t, n), (n = n.sibling);
}
function wm(e, t, n) {
	if (Tt && typeof Tt.onCommitFiberUnmount == 'function')
		try {
			Tt.onCommitFiberUnmount(gs, n);
		} catch {}
	switch (n.tag) {
		case 5:
			Ae || ur(n, t);
		case 6:
			var r = ke,
				o = ct;
			(ke = null), Kt(e, t, n), (ke = r), (ct = o), ke !== null && (ct ? ((e = ke), (n = n.stateNode), e.nodeType === 8 ? e.parentNode.removeChild(n) : e.removeChild(n)) : ke.removeChild(n.stateNode));
			break;
		case 18:
			ke !== null && (ct ? ((e = ke), (n = n.stateNode), e.nodeType === 8 ? yl(e.parentNode, n) : e.nodeType === 1 && yl(e, n), yo(e)) : yl(ke, n.stateNode));
			break;
		case 4:
			(r = ke), (o = ct), (ke = n.stateNode.containerInfo), (ct = !0), Kt(e, t, n), (ke = r), (ct = o);
			break;
		case 0:
		case 11:
		case 14:
		case 15:
			if (!Ae && ((r = n.updateQueue), r !== null && ((r = r.lastEffect), r !== null))) {
				o = r = r.next;
				do {
					var i = o,
						s = i.destroy;
					(i = i.tag), s !== void 0 && (i & 2 || i & 4) && Ta(n, t, s), (o = o.next);
				} while (o !== r);
			}
			Kt(e, t, n);
			break;
		case 1:
			if (!Ae && (ur(n, t), (r = n.stateNode), typeof r.componentWillUnmount == 'function'))
				try {
					(r.props = n.memoizedProps), (r.state = n.memoizedState), r.componentWillUnmount();
				} catch (l) {
					fe(n, t, l);
				}
			Kt(e, t, n);
			break;
		case 21:
			Kt(e, t, n);
			break;
		case 22:
			n.mode & 1 ? ((Ae = (r = Ae) || n.memoizedState !== null), Kt(e, t, n), (Ae = r)) : Kt(e, t, n);
			break;
		default:
			Kt(e, t, n);
	}
}
function _d(e) {
	var t = e.updateQueue;
	if (t !== null) {
		e.updateQueue = null;
		var n = e.stateNode;
		n === null && (n = e.stateNode = new H0()),
			t.forEach(function (r) {
				var o = J0.bind(null, e, r);
				n.has(r) || (n.add(r), r.then(o, o));
			});
	}
}
function at(e, t) {
	var n = t.deletions;
	if (n !== null)
		for (var r = 0; r < n.length; r++) {
			var o = n[r];
			try {
				var i = e,
					s = t,
					l = s;
				e: for (; l !== null; ) {
					switch (l.tag) {
						case 5:
							(ke = l.stateNode), (ct = !1);
							break e;
						case 3:
							(ke = l.stateNode.containerInfo), (ct = !0);
							break e;
						case 4:
							(ke = l.stateNode.containerInfo), (ct = !0);
							break e;
					}
					l = l.return;
				}
				if (ke === null) throw Error(_(160));
				wm(i, s, o), (ke = null), (ct = !1);
				var a = o.alternate;
				a !== null && (a.return = null), (o.return = null);
			} catch (u) {
				fe(o, t, u);
			}
		}
	if (t.subtreeFlags & 12854) for (t = t.child; t !== null; ) xm(t, e), (t = t.sibling);
}
function xm(e, t) {
	var n = e.alternate,
		r = e.flags;
	switch (e.tag) {
		case 0:
		case 11:
		case 14:
		case 15:
			if ((at(t, e), Et(e), r & 4)) {
				try {
					uo(3, e, e.return), bs(3, e);
				} catch (y) {
					fe(e, e.return, y);
				}
				try {
					uo(5, e, e.return);
				} catch (y) {
					fe(e, e.return, y);
				}
			}
			break;
		case 1:
			at(t, e), Et(e), r & 512 && n !== null && ur(n, n.return);
			break;
		case 5:
			if ((at(t, e), Et(e), r & 512 && n !== null && ur(n, n.return), e.flags & 32)) {
				var o = e.stateNode;
				try {
					mo(o, '');
				} catch (y) {
					fe(e, e.return, y);
				}
			}
			if (r & 4 && ((o = e.stateNode), o != null)) {
				var i = e.memoizedProps,
					s = n !== null ? n.memoizedProps : i,
					l = e.type,
					a = e.updateQueue;
				if (((e.updateQueue = null), a !== null))
					try {
						l === 'input' && i.type === 'radio' && i.name != null && Uf(o, i), Jl(l, s);
						var u = Jl(l, i);
						for (s = 0; s < a.length; s += 2) {
							var f = a[s],
								d = a[s + 1];
							f === 'style' ? Kf(o, d) : f === 'dangerouslySetInnerHTML' ? Vf(o, d) : f === 'children' ? mo(o, d) : iu(o, f, d, u);
						}
						switch (l) {
							case 'input':
								Yl(o, i);
								break;
							case 'textarea':
								Hf(o, i);
								break;
							case 'select':
								var c = o._wrapperState.wasMultiple;
								o._wrapperState.wasMultiple = !!i.multiple;
								var g = i.value;
								g != null ? dr(o, !!i.multiple, g, !1) : c !== !!i.multiple && (i.defaultValue != null ? dr(o, !!i.multiple, i.defaultValue, !0) : dr(o, !!i.multiple, i.multiple ? [] : '', !1));
						}
						o[Co] = i;
					} catch (y) {
						fe(e, e.return, y);
					}
			}
			break;
		case 6:
			if ((at(t, e), Et(e), r & 4)) {
				if (e.stateNode === null) throw Error(_(162));
				(o = e.stateNode), (i = e.memoizedProps);
				try {
					o.nodeValue = i;
				} catch (y) {
					fe(e, e.return, y);
				}
			}
			break;
		case 3:
			if ((at(t, e), Et(e), r & 4 && n !== null && n.memoizedState.isDehydrated))
				try {
					yo(t.containerInfo);
				} catch (y) {
					fe(e, e.return, y);
				}
			break;
		case 4:
			at(t, e), Et(e);
			break;
		case 13:
			at(t, e), Et(e), (o = e.child), o.flags & 8192 && ((i = o.memoizedState !== null), (o.stateNode.isHidden = i), !i || (o.alternate !== null && o.alternate.memoizedState !== null) || (Fu = me())), r & 4 && _d(e);
			break;
		case 22:
			if (((f = n !== null && n.memoizedState !== null), e.mode & 1 ? ((Ae = (u = Ae) || f), at(t, e), (Ae = u)) : at(t, e), Et(e), r & 8192)) {
				if (((u = e.memoizedState !== null), (e.stateNode.isHidden = u) && !f && e.mode & 1))
					for (I = e, f = e.child; f !== null; ) {
						for (d = I = f; I !== null; ) {
							switch (((c = I), (g = c.child), c.tag)) {
								case 0:
								case 11:
								case 14:
								case 15:
									uo(4, c, c.return);
									break;
								case 1:
									ur(c, c.return);
									var w = c.stateNode;
									if (typeof w.componentWillUnmount == 'function') {
										(r = c), (n = c.return);
										try {
											(t = r), (w.props = t.memoizedProps), (w.state = t.memoizedState), w.componentWillUnmount();
										} catch (y) {
											fe(r, n, y);
										}
									}
									break;
								case 5:
									ur(c, c.return);
									break;
								case 22:
									if (c.memoizedState !== null) {
										Ad(d);
										continue;
									}
							}
							g !== null ? ((g.return = c), (I = g)) : Ad(d);
						}
						f = f.sibling;
					}
				e: for (f = null, d = e; ; ) {
					if (d.tag === 5) {
						if (f === null) {
							f = d;
							try {
								(o = d.stateNode),
									u
										? ((i = o.style), typeof i.setProperty == 'function' ? i.setProperty('display', 'none', 'important') : (i.display = 'none'))
										: ((l = d.stateNode), (a = d.memoizedProps.style), (s = a != null && a.hasOwnProperty('display') ? a.display : null), (l.style.display = Gf('display', s)));
							} catch (y) {
								fe(e, e.return, y);
							}
						}
					} else if (d.tag === 6) {
						if (f === null)
							try {
								d.stateNode.nodeValue = u ? '' : d.memoizedProps;
							} catch (y) {
								fe(e, e.return, y);
							}
					} else if (((d.tag !== 22 && d.tag !== 23) || d.memoizedState === null || d === e) && d.child !== null) {
						(d.child.return = d), (d = d.child);
						continue;
					}
					if (d === e) break e;
					for (; d.sibling === null; ) {
						if (d.return === null || d.return === e) break e;
						f === d && (f = null), (d = d.return);
					}
					f === d && (f = null), (d.sibling.return = d.return), (d = d.sibling);
				}
			}
			break;
		case 19:
			at(t, e), Et(e), r & 4 && _d(e);
			break;
		case 21:
			break;
		default:
			at(t, e), Et(e);
	}
}
function Et(e) {
	var t = e.flags;
	if (t & 2) {
		try {
			e: {
				for (var n = e.return; n !== null; ) {
					if (ym(n)) {
						var r = n;
						break e;
					}
					n = n.return;
				}
				throw Error(_(160));
			}
			switch (r.tag) {
				case 5:
					var o = r.stateNode;
					r.flags & 32 && (mo(o, ''), (r.flags &= -33));
					var i = Nd(e);
					_a(e, i, o);
					break;
				case 3:
				case 4:
					var s = r.stateNode.containerInfo,
						l = Nd(e);
					Na(e, l, s);
					break;
				default:
					throw Error(_(161));
			}
		} catch (a) {
			fe(e, e.return, a);
		}
		e.flags &= -3;
	}
	t & 4096 && (e.flags &= -4097);
}
function V0(e, t, n) {
	(I = e), Sm(e);
}
function Sm(e, t, n) {
	for (var r = (e.mode & 1) !== 0; I !== null; ) {
		var o = I,
			i = o.child;
		if (o.tag === 22 && r) {
			var s = o.memoizedState !== null || ci;
			if (!s) {
				var l = o.alternate,
					a = (l !== null && l.memoizedState !== null) || Ae;
				l = ci;
				var u = Ae;
				if (((ci = s), (Ae = a) && !u)) for (I = o; I !== null; ) (s = I), (a = s.child), s.tag === 22 && s.memoizedState !== null ? Md(o) : a !== null ? ((a.return = s), (I = a)) : Md(o);
				for (; i !== null; ) (I = i), Sm(i), (i = i.sibling);
				(I = o), (ci = l), (Ae = u);
			}
			Rd(e);
		} else o.subtreeFlags & 8772 && i !== null ? ((i.return = o), (I = i)) : Rd(e);
	}
}
function Rd(e) {
	for (; I !== null; ) {
		var t = I;
		if (t.flags & 8772) {
			var n = t.alternate;
			try {
				if (t.flags & 8772)
					switch (t.tag) {
						case 0:
						case 11:
						case 15:
							Ae || bs(5, t);
							break;
						case 1:
							var r = t.stateNode;
							if (t.flags & 4 && !Ae)
								if (n === null) r.componentDidMount();
								else {
									var o = t.elementType === t.type ? n.memoizedProps : ut(t.type, n.memoizedProps);
									r.componentDidUpdate(o, n.memoizedState, r.__reactInternalSnapshotBeforeUpdate);
								}
							var i = t.updateQueue;
							i !== null && md(t, i, r);
							break;
						case 3:
							var s = t.updateQueue;
							if (s !== null) {
								if (((n = null), t.child !== null))
									switch (t.child.tag) {
										case 5:
											n = t.child.stateNode;
											break;
										case 1:
											n = t.child.stateNode;
									}
								md(t, s, n);
							}
							break;
						case 5:
							var l = t.stateNode;
							if (n === null && t.flags & 4) {
								n = l;
								var a = t.memoizedProps;
								switch (t.type) {
									case 'button':
									case 'input':
									case 'select':
									case 'textarea':
										a.autoFocus && n.focus();
										break;
									case 'img':
										a.src && (n.src = a.src);
								}
							}
							break;
						case 6:
							break;
						case 4:
							break;
						case 12:
							break;
						case 13:
							if (t.memoizedState === null) {
								var u = t.alternate;
								if (u !== null) {
									var f = u.memoizedState;
									if (f !== null) {
										var d = f.dehydrated;
										d !== null && yo(d);
									}
								}
							}
							break;
						case 19:
						case 17:
						case 21:
						case 22:
						case 23:
						case 25:
							break;
						default:
							throw Error(_(163));
					}
				Ae || (t.flags & 512 && Pa(t));
			} catch (c) {
				fe(t, t.return, c);
			}
		}
		if (t === e) {
			I = null;
			break;
		}
		if (((n = t.sibling), n !== null)) {
			(n.return = t.return), (I = n);
			break;
		}
		I = t.return;
	}
}
function Ad(e) {
	for (; I !== null; ) {
		var t = I;
		if (t === e) {
			I = null;
			break;
		}
		var n = t.sibling;
		if (n !== null) {
			(n.return = t.return), (I = n);
			break;
		}
		I = t.return;
	}
}
function Md(e) {
	for (; I !== null; ) {
		var t = I;
		try {
			switch (t.tag) {
				case 0:
				case 11:
				case 15:
					var n = t.return;
					try {
						bs(4, t);
					} catch (a) {
						fe(t, n, a);
					}
					break;
				case 1:
					var r = t.stateNode;
					if (typeof r.componentDidMount == 'function') {
						var o = t.return;
						try {
							r.componentDidMount();
						} catch (a) {
							fe(t, o, a);
						}
					}
					var i = t.return;
					try {
						Pa(t);
					} catch (a) {
						fe(t, i, a);
					}
					break;
				case 5:
					var s = t.return;
					try {
						Pa(t);
					} catch (a) {
						fe(t, s, a);
					}
			}
		} catch (a) {
			fe(t, t.return, a);
		}
		if (t === e) {
			I = null;
			break;
		}
		var l = t.sibling;
		if (l !== null) {
			(l.return = t.return), (I = l);
			break;
		}
		I = t.return;
	}
}
var G0 = Math.ceil,
	ts = Ht.ReactCurrentDispatcher,
	Du = Ht.ReactCurrentOwner,
	nt = Ht.ReactCurrentBatchConfig,
	Z = 0,
	Ce = null,
	ge = null,
	Te = 0,
	Ve = 0,
	cr = wn(0),
	ye = 0,
	_o = null,
	jn = 0,
	ks = 0,
	Lu = 0,
	co = null,
	ze = null,
	Fu = 0,
	Pr = 1 / 0,
	Mt = null,
	ns = !1,
	Ra = null,
	un = null,
	di = !1,
	nn = null,
	rs = 0,
	fo = 0,
	Aa = null,
	_i = -1,
	Ri = 0;
function je() {
	return Z & 6 ? me() : _i !== -1 ? _i : (_i = me());
}
function cn(e) {
	return e.mode & 1 ? (Z & 2 && Te !== 0 ? Te & -Te : _0.transition !== null ? (Ri === 0 && (Ri = ip()), Ri) : ((e = q), e !== 0 || ((e = window.event), (e = e === void 0 ? 16 : fp(e.type))), e)) : 1;
}
function mt(e, t, n, r) {
	if (50 < fo) throw ((fo = 0), (Aa = null), Error(_(185)));
	Fo(e, n, r), (!(Z & 2) || e !== Ce) && (e === Ce && (!(Z & 2) && (ks |= n), ye === 4 && en(e, Te)), He(e, r), n === 1 && Z === 0 && !(t.mode & 1) && ((Pr = me() + 500), Ss && xn()));
}
function He(e, t) {
	var n = e.callbackNode;
	_y(e, t);
	var r = $i(e, e === Ce ? Te : 0);
	if (r === 0) n !== null && Bc(n), (e.callbackNode = null), (e.callbackPriority = 0);
	else if (((t = r & -r), e.callbackPriority !== t)) {
		if ((n != null && Bc(n), t === 1))
			e.tag === 0 ? N0(Od.bind(null, e)) : Ap(Od.bind(null, e)),
				b0(function () {
					!(Z & 6) && xn();
				}),
				(n = null);
		else {
			switch (sp(r)) {
				case 1:
					n = cu;
					break;
				case 4:
					n = rp;
					break;
				case 16:
					n = zi;
					break;
				case 536870912:
					n = op;
					break;
				default:
					n = zi;
			}
			n = _m(n, Em.bind(null, e));
		}
		(e.callbackPriority = t), (e.callbackNode = n);
	}
}
function Em(e, t) {
	if (((_i = -1), (Ri = 0), Z & 6)) throw Error(_(327));
	var n = e.callbackNode;
	if (gr() && e.callbackNode !== n) return null;
	var r = $i(e, e === Ce ? Te : 0);
	if (r === 0) return null;
	if (r & 30 || r & e.expiredLanes || t) t = os(e, r);
	else {
		t = r;
		var o = Z;
		Z |= 2;
		var i = bm();
		(Ce !== e || Te !== t) && ((Mt = null), (Pr = me() + 500), Pn(e, t));
		do
			try {
				Q0();
				break;
			} catch (l) {
				Cm(e, l);
			}
		while (!0);
		Cu(), (ts.current = i), (Z = o), ge !== null ? (t = 0) : ((Ce = null), (Te = 0), (t = ye));
	}
	if (t !== 0) {
		if ((t === 2 && ((o = oa(e)), o !== 0 && ((r = o), (t = Ma(e, o)))), t === 1)) throw ((n = _o), Pn(e, 0), en(e, r), He(e, me()), n);
		if (t === 6) en(e, r);
		else {
			if (((o = e.current.alternate), !(r & 30) && !K0(o) && ((t = os(e, r)), t === 2 && ((i = oa(e)), i !== 0 && ((r = i), (t = Ma(e, i)))), t === 1))) throw ((n = _o), Pn(e, 0), en(e, r), He(e, me()), n);
			switch (((e.finishedWork = o), (e.finishedLanes = r), t)) {
				case 0:
				case 1:
					throw Error(_(345));
				case 2:
					Cn(e, ze, Mt);
					break;
				case 3:
					if ((en(e, r), (r & 130023424) === r && ((t = Fu + 500 - me()), 10 < t))) {
						if ($i(e, 0) !== 0) break;
						if (((o = e.suspendedLanes), (o & r) !== r)) {
							je(), (e.pingedLanes |= e.suspendedLanes & o);
							break;
						}
						e.timeoutHandle = fa(Cn.bind(null, e, ze, Mt), t);
						break;
					}
					Cn(e, ze, Mt);
					break;
				case 4:
					if ((en(e, r), (r & 4194240) === r)) break;
					for (t = e.eventTimes, o = -1; 0 < r; ) {
						var s = 31 - pt(r);
						(i = 1 << s), (s = t[s]), s > o && (o = s), (r &= ~i);
					}
					if (((r = o), (r = me() - r), (r = (120 > r ? 120 : 480 > r ? 480 : 1080 > r ? 1080 : 1920 > r ? 1920 : 3e3 > r ? 3e3 : 4320 > r ? 4320 : 1960 * G0(r / 1960)) - r), 10 < r)) {
						e.timeoutHandle = fa(Cn.bind(null, e, ze, Mt), r);
						break;
					}
					Cn(e, ze, Mt);
					break;
				case 5:
					Cn(e, ze, Mt);
					break;
				default:
					throw Error(_(329));
			}
		}
	}
	return He(e, me()), e.callbackNode === n ? Em.bind(null, e) : null;
}
function Ma(e, t) {
	var n = co;
	return e.current.memoizedState.isDehydrated && (Pn(e, t).flags |= 256), (e = os(e, t)), e !== 2 && ((t = ze), (ze = n), t !== null && Oa(t)), e;
}
function Oa(e) {
	ze === null ? (ze = e) : ze.push.apply(ze, e);
}
function K0(e) {
	for (var t = e; ; ) {
		if (t.flags & 16384) {
			var n = t.updateQueue;
			if (n !== null && ((n = n.stores), n !== null))
				for (var r = 0; r < n.length; r++) {
					var o = n[r],
						i = o.getSnapshot;
					o = o.value;
					try {
						if (!ht(i(), o)) return !1;
					} catch {
						return !1;
					}
				}
		}
		if (((n = t.child), t.subtreeFlags & 16384 && n !== null)) (n.return = t), (t = n);
		else {
			if (t === e) break;
			for (; t.sibling === null; ) {
				if (t.return === null || t.return === e) return !0;
				t = t.return;
			}
			(t.sibling.return = t.return), (t = t.sibling);
		}
	}
	return !0;
}
function en(e, t) {
	for (t &= ~Lu, t &= ~ks, e.suspendedLanes |= t, e.pingedLanes &= ~t, e = e.expirationTimes; 0 < t; ) {
		var n = 31 - pt(t),
			r = 1 << n;
		(e[n] = -1), (t &= ~r);
	}
}
function Od(e) {
	if (Z & 6) throw Error(_(327));
	gr();
	var t = $i(e, 0);
	if (!(t & 1)) return He(e, me()), null;
	var n = os(e, t);
	if (e.tag !== 0 && n === 2) {
		var r = oa(e);
		r !== 0 && ((t = r), (n = Ma(e, r)));
	}
	if (n === 1) throw ((n = _o), Pn(e, 0), en(e, t), He(e, me()), n);
	if (n === 6) throw Error(_(345));
	return (e.finishedWork = e.current.alternate), (e.finishedLanes = t), Cn(e, ze, Mt), He(e, me()), null;
}
function zu(e, t) {
	var n = Z;
	Z |= 1;
	try {
		return e(t);
	} finally {
		(Z = n), Z === 0 && ((Pr = me() + 500), Ss && xn());
	}
}
function In(e) {
	nn !== null && nn.tag === 0 && !(Z & 6) && gr();
	var t = Z;
	Z |= 1;
	var n = nt.transition,
		r = q;
	try {
		if (((nt.transition = null), (q = 1), e)) return e();
	} finally {
		(q = r), (nt.transition = n), (Z = t), !(Z & 6) && xn();
	}
}
function $u() {
	(Ve = cr.current), oe(cr);
}
function Pn(e, t) {
	(e.finishedWork = null), (e.finishedLanes = 0);
	var n = e.timeoutHandle;
	if ((n !== -1 && ((e.timeoutHandle = -1), C0(n)), ge !== null))
		for (n = ge.return; n !== null; ) {
			var r = n;
			switch ((xu(r), r.tag)) {
				case 1:
					(r = r.type.childContextTypes), r != null && Vi();
					break;
				case 3:
					kr(), oe(Be), oe(Me), _u();
					break;
				case 5:
					Nu(r);
					break;
				case 4:
					kr();
					break;
				case 13:
					oe(ue);
					break;
				case 19:
					oe(ue);
					break;
				case 10:
					bu(r.type._context);
					break;
				case 22:
				case 23:
					$u();
			}
			n = n.return;
		}
	if (((Ce = e), (ge = e = dn(e.current, null)), (Te = Ve = t), (ye = 0), (_o = null), (Lu = ks = jn = 0), (ze = co = null), kn !== null)) {
		for (t = 0; t < kn.length; t++)
			if (((n = kn[t]), (r = n.interleaved), r !== null)) {
				n.interleaved = null;
				var o = r.next,
					i = n.pending;
				if (i !== null) {
					var s = i.next;
					(i.next = o), (r.next = s);
				}
				n.pending = r;
			}
		kn = null;
	}
	return e;
}
function Cm(e, t) {
	do {
		var n = ge;
		try {
			if ((Cu(), (Ti.current = es), Ji)) {
				for (var r = ce.memoizedState; r !== null; ) {
					var o = r.queue;
					o !== null && (o.pending = null), (r = r.next);
				}
				Ji = !1;
			}
			if (((On = 0), (Ee = ve = ce = null), (ao = !1), (To = 0), (Du.current = null), n === null || n.return === null)) {
				(ye = 1), (_o = t), (ge = null);
				break;
			}
			e: {
				var i = e,
					s = n.return,
					l = n,
					a = t;
				if (((t = Te), (l.flags |= 32768), a !== null && typeof a == 'object' && typeof a.then == 'function')) {
					var u = a,
						f = l,
						d = f.tag;
					if (!(f.mode & 1) && (d === 0 || d === 11 || d === 15)) {
						var c = f.alternate;
						c ? ((f.updateQueue = c.updateQueue), (f.memoizedState = c.memoizedState), (f.lanes = c.lanes)) : ((f.updateQueue = null), (f.memoizedState = null));
					}
					var g = xd(s);
					if (g !== null) {
						(g.flags &= -257), Sd(g, s, l, i, t), g.mode & 1 && wd(i, u, t), (t = g), (a = u);
						var w = t.updateQueue;
						if (w === null) {
							var y = new Set();
							y.add(a), (t.updateQueue = y);
						} else w.add(a);
						break e;
					} else {
						if (!(t & 1)) {
							wd(i, u, t), Bu();
							break e;
						}
						a = Error(_(426));
					}
				} else if (ae && l.mode & 1) {
					var S = xd(s);
					if (S !== null) {
						!(S.flags & 65536) && (S.flags |= 256), Sd(S, s, l, i, t), Su(Tr(a, l));
						break e;
					}
				}
				(i = a = Tr(a, l)), ye !== 4 && (ye = 2), co === null ? (co = [i]) : co.push(i), (i = s);
				do {
					switch (i.tag) {
						case 3:
							(i.flags |= 65536), (t &= -t), (i.lanes |= t);
							var m = sm(i, a, t);
							pd(i, m);
							break e;
						case 1:
							l = a;
							var h = i.type,
								x = i.stateNode;
							if (!(i.flags & 128) && (typeof h.getDerivedStateFromError == 'function' || (x !== null && typeof x.componentDidCatch == 'function' && (un === null || !un.has(x))))) {
								(i.flags |= 65536), (t &= -t), (i.lanes |= t);
								var E = lm(i, l, t);
								pd(i, E);
								break e;
							}
					}
					i = i.return;
				} while (i !== null);
			}
			Tm(n);
		} catch (C) {
			(t = C), ge === n && n !== null && (ge = n = n.return);
			continue;
		}
		break;
	} while (!0);
}
function bm() {
	var e = ts.current;
	return (ts.current = es), e === null ? es : e;
}
function Bu() {
	(ye === 0 || ye === 3 || ye === 2) && (ye = 4), Ce === null || (!(jn & 268435455) && !(ks & 268435455)) || en(Ce, Te);
}
function os(e, t) {
	var n = Z;
	Z |= 2;
	var r = bm();
	(Ce !== e || Te !== t) && ((Mt = null), Pn(e, t));
	do
		try {
			Y0();
			break;
		} catch (o) {
			Cm(e, o);
		}
	while (!0);
	if ((Cu(), (Z = n), (ts.current = r), ge !== null)) throw Error(_(261));
	return (Ce = null), (Te = 0), ye;
}
function Y0() {
	for (; ge !== null; ) km(ge);
}
function Q0() {
	for (; ge !== null && !xy(); ) km(ge);
}
function km(e) {
	var t = Nm(e.alternate, e, Ve);
	(e.memoizedProps = e.pendingProps), t === null ? Tm(e) : (ge = t), (Du.current = null);
}
function Tm(e) {
	var t = e;
	do {
		var n = t.alternate;
		if (((e = t.return), t.flags & 32768)) {
			if (((n = U0(n, t)), n !== null)) {
				(n.flags &= 32767), (ge = n);
				return;
			}
			if (e !== null) (e.flags |= 32768), (e.subtreeFlags = 0), (e.deletions = null);
			else {
				(ye = 6), (ge = null);
				return;
			}
		} else if (((n = B0(n, t, Ve)), n !== null)) {
			ge = n;
			return;
		}
		if (((t = t.sibling), t !== null)) {
			ge = t;
			return;
		}
		ge = t = e;
	} while (t !== null);
	ye === 0 && (ye = 5);
}
function Cn(e, t, n) {
	var r = q,
		o = nt.transition;
	try {
		(nt.transition = null), (q = 1), X0(e, t, n, r);
	} finally {
		(nt.transition = o), (q = r);
	}
	return null;
}
function X0(e, t, n, r) {
	do gr();
	while (nn !== null);
	if (Z & 6) throw Error(_(327));
	n = e.finishedWork;
	var o = e.finishedLanes;
	if (n === null) return null;
	if (((e.finishedWork = null), (e.finishedLanes = 0), n === e.current)) throw Error(_(177));
	(e.callbackNode = null), (e.callbackPriority = 0);
	var i = n.lanes | n.childLanes;
	if (
		(Ry(e, i),
		e === Ce && ((ge = Ce = null), (Te = 0)),
		(!(n.subtreeFlags & 2064) && !(n.flags & 2064)) ||
			di ||
			((di = !0),
			_m(zi, function () {
				return gr(), null;
			})),
		(i = (n.flags & 15990) !== 0),
		n.subtreeFlags & 15990 || i)
	) {
		(i = nt.transition), (nt.transition = null);
		var s = q;
		q = 1;
		var l = Z;
		(Z |= 4), (Du.current = null), W0(e, n), xm(n, e), g0(ca), (Bi = !!ua), (ca = ua = null), (e.current = n), V0(n), Sy(), (Z = l), (q = s), (nt.transition = i);
	} else e.current = n;
	if ((di && ((di = !1), (nn = e), (rs = o)), (i = e.pendingLanes), i === 0 && (un = null), by(n.stateNode), He(e, me()), t !== null))
		for (r = e.onRecoverableError, n = 0; n < t.length; n++) (o = t[n]), r(o.value, {componentStack: o.stack, digest: o.digest});
	if (ns) throw ((ns = !1), (e = Ra), (Ra = null), e);
	return rs & 1 && e.tag !== 0 && gr(), (i = e.pendingLanes), i & 1 ? (e === Aa ? fo++ : ((fo = 0), (Aa = e))) : (fo = 0), xn(), null;
}
function gr() {
	if (nn !== null) {
		var e = sp(rs),
			t = nt.transition,
			n = q;
		try {
			if (((nt.transition = null), (q = 16 > e ? 16 : e), nn === null)) var r = !1;
			else {
				if (((e = nn), (nn = null), (rs = 0), Z & 6)) throw Error(_(331));
				var o = Z;
				for (Z |= 4, I = e.current; I !== null; ) {
					var i = I,
						s = i.child;
					if (I.flags & 16) {
						var l = i.deletions;
						if (l !== null) {
							for (var a = 0; a < l.length; a++) {
								var u = l[a];
								for (I = u; I !== null; ) {
									var f = I;
									switch (f.tag) {
										case 0:
										case 11:
										case 15:
											uo(8, f, i);
									}
									var d = f.child;
									if (d !== null) (d.return = f), (I = d);
									else
										for (; I !== null; ) {
											f = I;
											var c = f.sibling,
												g = f.return;
											if ((vm(f), f === u)) {
												I = null;
												break;
											}
											if (c !== null) {
												(c.return = g), (I = c);
												break;
											}
											I = g;
										}
								}
							}
							var w = i.alternate;
							if (w !== null) {
								var y = w.child;
								if (y !== null) {
									w.child = null;
									do {
										var S = y.sibling;
										(y.sibling = null), (y = S);
									} while (y !== null);
								}
							}
							I = i;
						}
					}
					if (i.subtreeFlags & 2064 && s !== null) (s.return = i), (I = s);
					else
						e: for (; I !== null; ) {
							if (((i = I), i.flags & 2048))
								switch (i.tag) {
									case 0:
									case 11:
									case 15:
										uo(9, i, i.return);
								}
							var m = i.sibling;
							if (m !== null) {
								(m.return = i.return), (I = m);
								break e;
							}
							I = i.return;
						}
				}
				var h = e.current;
				for (I = h; I !== null; ) {
					s = I;
					var x = s.child;
					if (s.subtreeFlags & 2064 && x !== null) (x.return = s), (I = x);
					else
						e: for (s = h; I !== null; ) {
							if (((l = I), l.flags & 2048))
								try {
									switch (l.tag) {
										case 0:
										case 11:
										case 15:
											bs(9, l);
									}
								} catch (C) {
									fe(l, l.return, C);
								}
							if (l === s) {
								I = null;
								break e;
							}
							var E = l.sibling;
							if (E !== null) {
								(E.return = l.return), (I = E);
								break e;
							}
							I = l.return;
						}
				}
				if (((Z = o), xn(), Tt && typeof Tt.onPostCommitFiberRoot == 'function'))
					try {
						Tt.onPostCommitFiberRoot(gs, e);
					} catch {}
				r = !0;
			}
			return r;
		} finally {
			(q = n), (nt.transition = t);
		}
	}
	return !1;
}
function jd(e, t, n) {
	(t = Tr(n, t)), (t = sm(e, t, 1)), (e = an(e, t, 1)), (t = je()), e !== null && (Fo(e, 1, t), He(e, t));
}
function fe(e, t, n) {
	if (e.tag === 3) jd(e, e, n);
	else
		for (; t !== null; ) {
			if (t.tag === 3) {
				jd(t, e, n);
				break;
			} else if (t.tag === 1) {
				var r = t.stateNode;
				if (typeof t.type.getDerivedStateFromError == 'function' || (typeof r.componentDidCatch == 'function' && (un === null || !un.has(r)))) {
					(e = Tr(n, e)), (e = lm(t, e, 1)), (t = an(t, e, 1)), (e = je()), t !== null && (Fo(t, 1, e), He(t, e));
					break;
				}
			}
			t = t.return;
		}
}
function Z0(e, t, n) {
	var r = e.pingCache;
	r !== null && r.delete(t), (t = je()), (e.pingedLanes |= e.suspendedLanes & n), Ce === e && (Te & n) === n && (ye === 4 || (ye === 3 && (Te & 130023424) === Te && 500 > me() - Fu) ? Pn(e, 0) : (Lu |= n)), He(e, t);
}
function Pm(e, t) {
	t === 0 && (e.mode & 1 ? ((t = ti), (ti <<= 1), !(ti & 130023424) && (ti = 4194304)) : (t = 1));
	var n = je();
	(e = zt(e, t)), e !== null && (Fo(e, t, n), He(e, n));
}
function q0(e) {
	var t = e.memoizedState,
		n = 0;
	t !== null && (n = t.retryLane), Pm(e, n);
}
function J0(e, t) {
	var n = 0;
	switch (e.tag) {
		case 13:
			var r = e.stateNode,
				o = e.memoizedState;
			o !== null && (n = o.retryLane);
			break;
		case 19:
			r = e.stateNode;
			break;
		default:
			throw Error(_(314));
	}
	r !== null && r.delete(t), Pm(e, n);
}
var Nm;
Nm = function (e, t, n) {
	if (e !== null)
		if (e.memoizedProps !== t.pendingProps || Be.current) $e = !0;
		else {
			if (!(e.lanes & n) && !(t.flags & 128)) return ($e = !1), $0(e, t, n);
			$e = !!(e.flags & 131072);
		}
	else ($e = !1), ae && t.flags & 1048576 && Mp(t, Yi, t.index);
	switch (((t.lanes = 0), t.tag)) {
		case 2:
			var r = t.type;
			Ni(e, t), (e = t.pendingProps);
			var o = Er(t, Me.current);
			hr(t, n), (o = Au(null, t, r, e, o, n));
			var i = Mu();
			return (
				(t.flags |= 1),
				typeof o == 'object' && o !== null && typeof o.render == 'function' && o.$$typeof === void 0
					? ((t.tag = 1),
					  (t.memoizedState = null),
					  (t.updateQueue = null),
					  Ue(r) ? ((i = !0), Gi(t)) : (i = !1),
					  (t.memoizedState = o.state !== null && o.state !== void 0 ? o.state : null),
					  Tu(t),
					  (o.updater = Cs),
					  (t.stateNode = o),
					  (o._reactInternals = t),
					  wa(t, r, e, n),
					  (t = Ea(null, t, r, !0, i, n)))
					: ((t.tag = 0), ae && i && wu(t), Oe(null, t, o, n), (t = t.child)),
				t
			);
		case 16:
			r = t.elementType;
			e: {
				switch ((Ni(e, t), (e = t.pendingProps), (o = r._init), (r = o(r._payload)), (t.type = r), (o = t.tag = tw(r)), (e = ut(r, e)), o)) {
					case 0:
						t = Sa(null, t, r, e, n);
						break e;
					case 1:
						t = bd(null, t, r, e, n);
						break e;
					case 11:
						t = Ed(null, t, r, e, n);
						break e;
					case 14:
						t = Cd(null, t, r, ut(r.type, e), n);
						break e;
				}
				throw Error(_(306, r, ''));
			}
			return t;
		case 0:
			return (r = t.type), (o = t.pendingProps), (o = t.elementType === r ? o : ut(r, o)), Sa(e, t, r, o, n);
		case 1:
			return (r = t.type), (o = t.pendingProps), (o = t.elementType === r ? o : ut(r, o)), bd(e, t, r, o, n);
		case 3:
			e: {
				if ((dm(t), e === null)) throw Error(_(387));
				(r = t.pendingProps), (i = t.memoizedState), (o = i.element), Fp(e, t), Zi(t, r, null, n);
				var s = t.memoizedState;
				if (((r = s.element), i.isDehydrated))
					if (((i = {element: r, isDehydrated: !1, cache: s.cache, pendingSuspenseBoundaries: s.pendingSuspenseBoundaries, transitions: s.transitions}), (t.updateQueue.baseState = i), (t.memoizedState = i), t.flags & 256)) {
						(o = Tr(Error(_(423)), t)), (t = kd(e, t, r, n, o));
						break e;
					} else if (r !== o) {
						(o = Tr(Error(_(424)), t)), (t = kd(e, t, r, n, o));
						break e;
					} else for (Ke = ln(t.stateNode.containerInfo.firstChild), Ye = t, ae = !0, dt = null, n = Dp(t, null, r, n), t.child = n; n; ) (n.flags = (n.flags & -3) | 4096), (n = n.sibling);
				else {
					if ((Cr(), r === o)) {
						t = $t(e, t, n);
						break e;
					}
					Oe(e, t, r, n);
				}
				t = t.child;
			}
			return t;
		case 5:
			return zp(t), e === null && ga(t), (r = t.type), (o = t.pendingProps), (i = e !== null ? e.memoizedProps : null), (s = o.children), da(r, o) ? (s = null) : i !== null && da(r, i) && (t.flags |= 32), cm(e, t), Oe(e, t, s, n), t.child;
		case 6:
			return e === null && ga(t), null;
		case 13:
			return fm(e, t, n);
		case 4:
			return Pu(t, t.stateNode.containerInfo), (r = t.pendingProps), e === null ? (t.child = br(t, null, r, n)) : Oe(e, t, r, n), t.child;
		case 11:
			return (r = t.type), (o = t.pendingProps), (o = t.elementType === r ? o : ut(r, o)), Ed(e, t, r, o, n);
		case 7:
			return Oe(e, t, t.pendingProps, n), t.child;
		case 8:
			return Oe(e, t, t.pendingProps.children, n), t.child;
		case 12:
			return Oe(e, t, t.pendingProps.children, n), t.child;
		case 10:
			e: {
				if (((r = t.type._context), (o = t.pendingProps), (i = t.memoizedProps), (s = o.value), ee(Qi, r._currentValue), (r._currentValue = s), i !== null))
					if (ht(i.value, s)) {
						if (i.children === o.children && !Be.current) {
							t = $t(e, t, n);
							break e;
						}
					} else
						for (i = t.child, i !== null && (i.return = t); i !== null; ) {
							var l = i.dependencies;
							if (l !== null) {
								s = i.child;
								for (var a = l.firstContext; a !== null; ) {
									if (a.context === r) {
										if (i.tag === 1) {
											(a = Dt(-1, n & -n)), (a.tag = 2);
											var u = i.updateQueue;
											if (u !== null) {
												u = u.shared;
												var f = u.pending;
												f === null ? (a.next = a) : ((a.next = f.next), (f.next = a)), (u.pending = a);
											}
										}
										(i.lanes |= n), (a = i.alternate), a !== null && (a.lanes |= n), va(i.return, n, t), (l.lanes |= n);
										break;
									}
									a = a.next;
								}
							} else if (i.tag === 10) s = i.type === t.type ? null : i.child;
							else if (i.tag === 18) {
								if (((s = i.return), s === null)) throw Error(_(341));
								(s.lanes |= n), (l = s.alternate), l !== null && (l.lanes |= n), va(s, n, t), (s = i.sibling);
							} else s = i.child;
							if (s !== null) s.return = i;
							else
								for (s = i; s !== null; ) {
									if (s === t) {
										s = null;
										break;
									}
									if (((i = s.sibling), i !== null)) {
										(i.return = s.return), (s = i);
										break;
									}
									s = s.return;
								}
							i = s;
						}
				Oe(e, t, o.children, n), (t = t.child);
			}
			return t;
		case 9:
			return (o = t.type), (r = t.pendingProps.children), hr(t, n), (o = ot(o)), (r = r(o)), (t.flags |= 1), Oe(e, t, r, n), t.child;
		case 14:
			return (r = t.type), (o = ut(r, t.pendingProps)), (o = ut(r.type, o)), Cd(e, t, r, o, n);
		case 15:
			return am(e, t, t.type, t.pendingProps, n);
		case 17:
			return (r = t.type), (o = t.pendingProps), (o = t.elementType === r ? o : ut(r, o)), Ni(e, t), (t.tag = 1), Ue(r) ? ((e = !0), Gi(t)) : (e = !1), hr(t, n), im(t, r, o), wa(t, r, o, n), Ea(null, t, r, !0, e, n);
		case 19:
			return pm(e, t, n);
		case 22:
			return um(e, t, n);
	}
	throw Error(_(156, t.tag));
};
function _m(e, t) {
	return np(e, t);
}
function ew(e, t, n, r) {
	(this.tag = e),
		(this.key = n),
		(this.sibling = this.child = this.return = this.stateNode = this.type = this.elementType = null),
		(this.index = 0),
		(this.ref = null),
		(this.pendingProps = t),
		(this.dependencies = this.memoizedState = this.updateQueue = this.memoizedProps = null),
		(this.mode = r),
		(this.subtreeFlags = this.flags = 0),
		(this.deletions = null),
		(this.childLanes = this.lanes = 0),
		(this.alternate = null);
}
function tt(e, t, n, r) {
	return new ew(e, t, n, r);
}
function Uu(e) {
	return (e = e.prototype), !(!e || !e.isReactComponent);
}
function tw(e) {
	if (typeof e == 'function') return Uu(e) ? 1 : 0;
	if (e != null) {
		if (((e = e.$$typeof), e === lu)) return 11;
		if (e === au) return 14;
	}
	return 2;
}
function dn(e, t) {
	var n = e.alternate;
	return (
		n === null
			? ((n = tt(e.tag, t, e.key, e.mode)), (n.elementType = e.elementType), (n.type = e.type), (n.stateNode = e.stateNode), (n.alternate = e), (e.alternate = n))
			: ((n.pendingProps = t), (n.type = e.type), (n.flags = 0), (n.subtreeFlags = 0), (n.deletions = null)),
		(n.flags = e.flags & 14680064),
		(n.childLanes = e.childLanes),
		(n.lanes = e.lanes),
		(n.child = e.child),
		(n.memoizedProps = e.memoizedProps),
		(n.memoizedState = e.memoizedState),
		(n.updateQueue = e.updateQueue),
		(t = e.dependencies),
		(n.dependencies = t === null ? null : {lanes: t.lanes, firstContext: t.firstContext}),
		(n.sibling = e.sibling),
		(n.index = e.index),
		(n.ref = e.ref),
		n
	);
}
function Ai(e, t, n, r, o, i) {
	var s = 2;
	if (((r = e), typeof e == 'function')) Uu(e) && (s = 1);
	else if (typeof e == 'string') s = 5;
	else
		e: switch (e) {
			case er:
				return Nn(n.children, o, i, t);
			case su:
				(s = 8), (o |= 8);
				break;
			case Hl:
				return (e = tt(12, n, t, o | 2)), (e.elementType = Hl), (e.lanes = i), e;
			case Wl:
				return (e = tt(13, n, t, o)), (e.elementType = Wl), (e.lanes = i), e;
			case Vl:
				return (e = tt(19, n, t, o)), (e.elementType = Vl), (e.lanes = i), e;
			case zf:
				return Ts(n, o, i, t);
			default:
				if (typeof e == 'object' && e !== null)
					switch (e.$$typeof) {
						case Lf:
							s = 10;
							break e;
						case Ff:
							s = 9;
							break e;
						case lu:
							s = 11;
							break e;
						case au:
							s = 14;
							break e;
						case Zt:
							(s = 16), (r = null);
							break e;
					}
				throw Error(_(130, e == null ? e : typeof e, ''));
		}
	return (t = tt(s, n, t, o)), (t.elementType = e), (t.type = r), (t.lanes = i), t;
}
function Nn(e, t, n, r) {
	return (e = tt(7, e, r, t)), (e.lanes = n), e;
}
function Ts(e, t, n, r) {
	return (e = tt(22, e, r, t)), (e.elementType = zf), (e.lanes = n), (e.stateNode = {isHidden: !1}), e;
}
function Tl(e, t, n) {
	return (e = tt(6, e, null, t)), (e.lanes = n), e;
}
function Pl(e, t, n) {
	return (t = tt(4, e.children !== null ? e.children : [], e.key, t)), (t.lanes = n), (t.stateNode = {containerInfo: e.containerInfo, pendingChildren: null, implementation: e.implementation}), t;
}
function nw(e, t, n, r, o) {
	(this.tag = t),
		(this.containerInfo = e),
		(this.finishedWork = this.pingCache = this.current = this.pendingChildren = null),
		(this.timeoutHandle = -1),
		(this.callbackNode = this.pendingContext = this.context = null),
		(this.callbackPriority = 0),
		(this.eventTimes = ll(0)),
		(this.expirationTimes = ll(-1)),
		(this.entangledLanes = this.finishedLanes = this.mutableReadLanes = this.expiredLanes = this.pingedLanes = this.suspendedLanes = this.pendingLanes = 0),
		(this.entanglements = ll(0)),
		(this.identifierPrefix = r),
		(this.onRecoverableError = o),
		(this.mutableSourceEagerHydrationData = null);
}
function Hu(e, t, n, r, o, i, s, l, a) {
	return (
		(e = new nw(e, t, n, l, a)),
		t === 1 ? ((t = 1), i === !0 && (t |= 8)) : (t = 0),
		(i = tt(3, null, null, t)),
		(e.current = i),
		(i.stateNode = e),
		(i.memoizedState = {element: r, isDehydrated: n, cache: null, transitions: null, pendingSuspenseBoundaries: null}),
		Tu(i),
		e
	);
}
function rw(e, t, n) {
	var r = 3 < arguments.length && arguments[3] !== void 0 ? arguments[3] : null;
	return {$$typeof: Jn, key: r == null ? null : '' + r, children: e, containerInfo: t, implementation: n};
}
function Rm(e) {
	if (!e) return pn;
	e = e._reactInternals;
	e: {
		if (Bn(e) !== e || e.tag !== 1) throw Error(_(170));
		var t = e;
		do {
			switch (t.tag) {
				case 3:
					t = t.stateNode.context;
					break e;
				case 1:
					if (Ue(t.type)) {
						t = t.stateNode.__reactInternalMemoizedMergedChildContext;
						break e;
					}
			}
			t = t.return;
		} while (t !== null);
		throw Error(_(171));
	}
	if (e.tag === 1) {
		var n = e.type;
		if (Ue(n)) return Rp(e, n, t);
	}
	return t;
}
function Am(e, t, n, r, o, i, s, l, a) {
	return (e = Hu(n, r, !0, e, o, i, s, l, a)), (e.context = Rm(null)), (n = e.current), (r = je()), (o = cn(n)), (i = Dt(r, o)), (i.callback = t ?? null), an(n, i, o), (e.current.lanes = o), Fo(e, o, r), He(e, r), e;
}
function Ps(e, t, n, r) {
	var o = t.current,
		i = je(),
		s = cn(o);
	return (
		(n = Rm(n)),
		t.context === null ? (t.context = n) : (t.pendingContext = n),
		(t = Dt(i, s)),
		(t.payload = {element: e}),
		(r = r === void 0 ? null : r),
		r !== null && (t.callback = r),
		(e = an(o, t, s)),
		e !== null && (mt(e, o, s, i), ki(e, o, s)),
		s
	);
}
function is(e) {
	if (((e = e.current), !e.child)) return null;
	switch (e.child.tag) {
		case 5:
			return e.child.stateNode;
		default:
			return e.child.stateNode;
	}
}
function Id(e, t) {
	if (((e = e.memoizedState), e !== null && e.dehydrated !== null)) {
		var n = e.retryLane;
		e.retryLane = n !== 0 && n < t ? n : t;
	}
}
function Wu(e, t) {
	Id(e, t), (e = e.alternate) && Id(e, t);
}
function ow() {
	return null;
}
var Mm =
	typeof reportError == 'function'
		? reportError
		: function (e) {
				console.error(e);
		  };
function Vu(e) {
	this._internalRoot = e;
}
Ns.prototype.render = Vu.prototype.render = function (e) {
	var t = this._internalRoot;
	if (t === null) throw Error(_(409));
	Ps(e, t, null, null);
};
Ns.prototype.unmount = Vu.prototype.unmount = function () {
	var e = this._internalRoot;
	if (e !== null) {
		this._internalRoot = null;
		var t = e.containerInfo;
		In(function () {
			Ps(null, e, null, null);
		}),
			(t[Ft] = null);
	}
};
function Ns(e) {
	this._internalRoot = e;
}
Ns.prototype.unstable_scheduleHydration = function (e) {
	if (e) {
		var t = up();
		e = {blockedOn: null, target: e, priority: t};
		for (var n = 0; n < Jt.length && t !== 0 && t < Jt[n].priority; n++);
		Jt.splice(n, 0, e), n === 0 && dp(e);
	}
};
function Gu(e) {
	return !(!e || (e.nodeType !== 1 && e.nodeType !== 9 && e.nodeType !== 11));
}
function _s(e) {
	return !(!e || (e.nodeType !== 1 && e.nodeType !== 9 && e.nodeType !== 11 && (e.nodeType !== 8 || e.nodeValue !== ' react-mount-point-unstable ')));
}
function Dd() {}
function iw(e, t, n, r, o) {
	if (o) {
		if (typeof r == 'function') {
			var i = r;
			r = function () {
				var u = is(s);
				i.call(u);
			};
		}
		var s = Am(t, r, e, 0, null, !1, !1, '', Dd);
		return (e._reactRootContainer = s), (e[Ft] = s.current), So(e.nodeType === 8 ? e.parentNode : e), In(), s;
	}
	for (; (o = e.lastChild); ) e.removeChild(o);
	if (typeof r == 'function') {
		var l = r;
		r = function () {
			var u = is(a);
			l.call(u);
		};
	}
	var a = Hu(e, 0, !1, null, null, !1, !1, '', Dd);
	return (
		(e._reactRootContainer = a),
		(e[Ft] = a.current),
		So(e.nodeType === 8 ? e.parentNode : e),
		In(function () {
			Ps(t, a, n, r);
		}),
		a
	);
}
function Rs(e, t, n, r, o) {
	var i = n._reactRootContainer;
	if (i) {
		var s = i;
		if (typeof o == 'function') {
			var l = o;
			o = function () {
				var a = is(s);
				l.call(a);
			};
		}
		Ps(t, s, e, o);
	} else s = iw(n, t, e, o, r);
	return is(s);
}
lp = function (e) {
	switch (e.tag) {
		case 3:
			var t = e.stateNode;
			if (t.current.memoizedState.isDehydrated) {
				var n = eo(t.pendingLanes);
				n !== 0 && (du(t, n | 1), He(t, me()), !(Z & 6) && ((Pr = me() + 500), xn()));
			}
			break;
		case 13:
			In(function () {
				var r = zt(e, 1);
				if (r !== null) {
					var o = je();
					mt(r, e, 1, o);
				}
			}),
				Wu(e, 1);
	}
};
fu = function (e) {
	if (e.tag === 13) {
		var t = zt(e, 134217728);
		if (t !== null) {
			var n = je();
			mt(t, e, 134217728, n);
		}
		Wu(e, 134217728);
	}
};
ap = function (e) {
	if (e.tag === 13) {
		var t = cn(e),
			n = zt(e, t);
		if (n !== null) {
			var r = je();
			mt(n, e, t, r);
		}
		Wu(e, t);
	}
};
up = function () {
	return q;
};
cp = function (e, t) {
	var n = q;
	try {
		return (q = e), t();
	} finally {
		q = n;
	}
};
ta = function (e, t, n) {
	switch (t) {
		case 'input':
			if ((Yl(e, n), (t = n.name), n.type === 'radio' && t != null)) {
				for (n = e; n.parentNode; ) n = n.parentNode;
				for (n = n.querySelectorAll('input[name=' + JSON.stringify('' + t) + '][type="radio"]'), t = 0; t < n.length; t++) {
					var r = n[t];
					if (r !== e && r.form === e.form) {
						var o = xs(r);
						if (!o) throw Error(_(90));
						Bf(r), Yl(r, o);
					}
				}
			}
			break;
		case 'textarea':
			Hf(e, n);
			break;
		case 'select':
			(t = n.value), t != null && dr(e, !!n.multiple, t, !1);
	}
};
Xf = zu;
Zf = In;
var sw = {usingClientEntryPoint: !1, Events: [$o, or, xs, Yf, Qf, zu]},
	Gr = {findFiberByHostInstance: bn, bundleType: 0, version: '18.3.1', rendererPackageName: 'react-dom'},
	lw = {
		bundleType: Gr.bundleType,
		version: Gr.version,
		rendererPackageName: Gr.rendererPackageName,
		rendererConfig: Gr.rendererConfig,
		overrideHookState: null,
		overrideHookStateDeletePath: null,
		overrideHookStateRenamePath: null,
		overrideProps: null,
		overridePropsDeletePath: null,
		overridePropsRenamePath: null,
		setErrorHandler: null,
		setSuspenseHandler: null,
		scheduleUpdate: null,
		currentDispatcherRef: Ht.ReactCurrentDispatcher,
		findHostInstanceByFiber: function (e) {
			return (e = ep(e)), e === null ? null : e.stateNode;
		},
		findFiberByHostInstance: Gr.findFiberByHostInstance || ow,
		findHostInstancesForRefresh: null,
		scheduleRefresh: null,
		scheduleRoot: null,
		setRefreshHandler: null,
		getCurrentFiber: null,
		reconcilerVersion: '18.3.1-next-f1338f8080-20240426',
	};
if (typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ < 'u') {
	var fi = __REACT_DEVTOOLS_GLOBAL_HOOK__;
	if (!fi.isDisabled && fi.supportsFiber)
		try {
			(gs = fi.inject(lw)), (Tt = fi);
		} catch {}
}
Ze.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED = sw;
Ze.createPortal = function (e, t) {
	var n = 2 < arguments.length && arguments[2] !== void 0 ? arguments[2] : null;
	if (!Gu(t)) throw Error(_(200));
	return rw(e, t, null, n);
};
Ze.createRoot = function (e, t) {
	if (!Gu(e)) throw Error(_(299));
	var n = !1,
		r = '',
		o = Mm;
	return (
		t != null && (t.unstable_strictMode === !0 && (n = !0), t.identifierPrefix !== void 0 && (r = t.identifierPrefix), t.onRecoverableError !== void 0 && (o = t.onRecoverableError)),
		(t = Hu(e, 1, !1, null, null, n, !1, r, o)),
		(e[Ft] = t.current),
		So(e.nodeType === 8 ? e.parentNode : e),
		new Vu(t)
	);
};
Ze.findDOMNode = function (e) {
	if (e == null) return null;
	if (e.nodeType === 1) return e;
	var t = e._reactInternals;
	if (t === void 0) throw typeof e.render == 'function' ? Error(_(188)) : ((e = Object.keys(e).join(',')), Error(_(268, e)));
	return (e = ep(t)), (e = e === null ? null : e.stateNode), e;
};
Ze.flushSync = function (e) {
	return In(e);
};
Ze.hydrate = function (e, t, n) {
	if (!_s(t)) throw Error(_(200));
	return Rs(null, e, t, !0, n);
};
Ze.hydrateRoot = function (e, t, n) {
	if (!Gu(e)) throw Error(_(405));
	var r = (n != null && n.hydratedSources) || null,
		o = !1,
		i = '',
		s = Mm;
	if (
		(n != null && (n.unstable_strictMode === !0 && (o = !0), n.identifierPrefix !== void 0 && (i = n.identifierPrefix), n.onRecoverableError !== void 0 && (s = n.onRecoverableError)),
		(t = Am(t, null, e, 1, n ?? null, o, !1, i, s)),
		(e[Ft] = t.current),
		So(e),
		r)
	)
		for (e = 0; e < r.length; e++) (n = r[e]), (o = n._getVersion), (o = o(n._source)), t.mutableSourceEagerHydrationData == null ? (t.mutableSourceEagerHydrationData = [n, o]) : t.mutableSourceEagerHydrationData.push(n, o);
	return new Ns(t);
};
Ze.render = function (e, t, n) {
	if (!_s(t)) throw Error(_(200));
	return Rs(null, e, t, !1, n);
};
Ze.unmountComponentAtNode = function (e) {
	if (!_s(e)) throw Error(_(40));
	return e._reactRootContainer
		? (In(function () {
				Rs(null, null, e, !1, function () {
					(e._reactRootContainer = null), (e[Ft] = null);
				});
		  }),
		  !0)
		: !1;
};
Ze.unstable_batchedUpdates = zu;
Ze.unstable_renderSubtreeIntoContainer = function (e, t, n, r) {
	if (!_s(n)) throw Error(_(200));
	if (e == null || e._reactInternals === void 0) throw Error(_(38));
	return Rs(e, t, n, !1, r);
};
Ze.version = '18.3.1-next-f1338f8080-20240426';
function Om() {
	if (!(typeof __REACT_DEVTOOLS_GLOBAL_HOOK__ > 'u' || typeof __REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE != 'function'))
		try {
			__REACT_DEVTOOLS_GLOBAL_HOOK__.checkDCE(Om);
		} catch (e) {
			console.error(e);
		}
}
Om(), (Of.exports = Ze);
var As = Of.exports;
const jm = Ar(As);
var Im,
	Ld = As;
(Im = Ld.createRoot), Ld.hydrateRoot;
const aw = 'modulepreload',
	uw = function (e) {
		return '/chat/' + e;
	},
	Fd = {},
	cw = function (t, n, r) {
		let o = Promise.resolve();
		if (n && n.length > 0) {
			document.getElementsByTagName('link');
			const s = document.querySelector('meta[property=csp-nonce]'),
				l = (s == null ? void 0 : s.nonce) || (s == null ? void 0 : s.getAttribute('nonce'));
			o = Promise.allSettled(
				n.map((a) => {
					if (((a = uw(a)), a in Fd)) return;
					Fd[a] = !0;
					const u = a.endsWith('.css'),
						f = u ? '[rel="stylesheet"]' : '';
					if (document.querySelector(`link[href="${a}"]${f}`)) return;
					const d = document.createElement('link');
					if (((d.rel = u ? 'stylesheet' : aw), u || (d.as = 'script'), (d.crossOrigin = ''), (d.href = a), l && d.setAttribute('nonce', l), document.head.appendChild(d), u))
						return new Promise((c, g) => {
							d.addEventListener('load', c), d.addEventListener('error', () => g(new Error(`Unable to preload CSS for ${a}`)));
						});
				})
			);
		}
		function i(s) {
			const l = new Event('vite:preloadError', {cancelable: !0});
			if (((l.payload = s), window.dispatchEvent(l), !l.defaultPrevented)) throw s;
		}
		return o.then((s) => {
			for (const l of s || []) l.status === 'rejected' && i(l.reason);
			return t().catch(i);
		});
	};
function dw(e) {
	const t = [];
	for (const n in e)
		if (Object.prototype.hasOwnProperty.call(e, n)) {
			const r = encodeURIComponent(`${e[n]}`);
			t.push(`${n}=${r}`);
		}
	return `?${t.join('&')}`;
}
const fw = 20,
	zd = 'Error: Gateway Timeout';
function Nl(e, t, n = [], r = !1, o = !0) {
	const [i, s] = p.useState(null),
		[l, a] = p.useState(!1),
		[u, f] = p.useState(null),
		[d, c] = p.useState(!1),
		g = t ? dw(t) : '';
	p.useEffect(() => {
		if (u && u.message !== zd) throw new Error(`Failed to fetch "${e}"`);
	}, [u, e]);
	const w = () => v.jsxs('div', {children: [v.jsx('div', {children: 'Something went wrong'}), v.jsx('div', {role: 'button', onClick: () => c((m) => !m), className: 'underline cursor-pointer', children: 'Click to retry'})]}),
		y = () => c((m) => !m);
	p.useEffect(() => {
		r && (u == null ? void 0 : u.message) === zd && (c((m) => !m), (u.message = ''));
	}, [r, u]);
	const S = p.useCallback(() => {
		const m = new AbortController(),
			{signal: h} = m;
		return (
			a(!0),
			f(null),
			fetch(`/${e}${g}`, {signal: h})
				.then(async (x) => {
					if (!x.ok) throw new Error(`Error: ${x.statusText}`);
					const E = await x.json();
					s(E);
				})
				.catch((x) => {
					x.code !== fw && f({message: x.message});
				})
				.finally(() => a(!1)),
			() => m.abort()
		);
	}, [e, d, ...n]);
	return (
		p.useEffect(() => {
			if (!o) return;
			const m = S();
			return () => {
				m();
			};
		}, [S, o]),
		{data: i, loading: l, error: u, refetch: y, ErrorTemplate: u && w}
	);
}
function Dm(e) {
	var t,
		n,
		r = '';
	if (typeof e == 'string' || typeof e == 'number') r += e;
	else if (typeof e == 'object')
		if (Array.isArray(e)) {
			var o = e.length;
			for (t = 0; t < o; t++) e[t] && (n = Dm(e[t])) && (r && (r += ' '), (r += n));
		} else for (n in e) e[n] && (r && (r += ' '), (r += n));
	return r;
}
function Ku() {
	for (var e, t, n = 0, r = '', o = arguments.length; n < o; n++) (e = arguments[n]) && (t = Dm(e)) && (r && (r += ' '), (r += t));
	return r;
}
const Yu = '-',
	pw = (e) => {
		const t = hw(e),
			{conflictingClassGroups: n, conflictingClassGroupModifiers: r} = e;
		return {
			getClassGroupId: (s) => {
				const l = s.split(Yu);
				return l[0] === '' && l.length !== 1 && l.shift(), Lm(l, t) || mw(s);
			},
			getConflictingClassGroupIds: (s, l) => {
				const a = n[s] || [];
				return l && r[s] ? [...a, ...r[s]] : a;
			},
		};
	},
	Lm = (e, t) => {
		var s;
		if (e.length === 0) return t.classGroupId;
		const n = e[0],
			r = t.nextPart.get(n),
			o = r ? Lm(e.slice(1), r) : void 0;
		if (o) return o;
		if (t.validators.length === 0) return;
		const i = e.join(Yu);
		return (s = t.validators.find(({validator: l}) => l(i))) == null ? void 0 : s.classGroupId;
	},
	$d = /^\[(.+)\]$/,
	mw = (e) => {
		if ($d.test(e)) {
			const t = $d.exec(e)[1],
				n = t == null ? void 0 : t.substring(0, t.indexOf(':'));
			if (n) return 'arbitrary..' + n;
		}
	},
	hw = (e) => {
		const {theme: t, prefix: n} = e,
			r = {nextPart: new Map(), validators: []};
		return (
			vw(Object.entries(e.classGroups), n).forEach(([i, s]) => {
				ja(s, r, i, t);
			}),
			r
		);
	},
	ja = (e, t, n, r) => {
		e.forEach((o) => {
			if (typeof o == 'string') {
				const i = o === '' ? t : Bd(t, o);
				i.classGroupId = n;
				return;
			}
			if (typeof o == 'function') {
				if (gw(o)) {
					ja(o(r), t, n, r);
					return;
				}
				t.validators.push({validator: o, classGroupId: n});
				return;
			}
			Object.entries(o).forEach(([i, s]) => {
				ja(s, Bd(t, i), n, r);
			});
		});
	},
	Bd = (e, t) => {
		let n = e;
		return (
			t.split(Yu).forEach((r) => {
				n.nextPart.has(r) || n.nextPart.set(r, {nextPart: new Map(), validators: []}), (n = n.nextPart.get(r));
			}),
			n
		);
	},
	gw = (e) => e.isThemeGetter,
	vw = (e, t) =>
		t
			? e.map(([n, r]) => {
					const o = r.map((i) => (typeof i == 'string' ? t + i : typeof i == 'object' ? Object.fromEntries(Object.entries(i).map(([s, l]) => [t + s, l])) : i));
					return [n, o];
			  })
			: e,
	yw = (e) => {
		if (e < 1) return {get: () => {}, set: () => {}};
		let t = 0,
			n = new Map(),
			r = new Map();
		const o = (i, s) => {
			n.set(i, s), t++, t > e && ((t = 0), (r = n), (n = new Map()));
		};
		return {
			get(i) {
				let s = n.get(i);
				if (s !== void 0) return s;
				if ((s = r.get(i)) !== void 0) return o(i, s), s;
			},
			set(i, s) {
				n.has(i) ? n.set(i, s) : o(i, s);
			},
		};
	},
	Fm = '!',
	ww = (e) => {
		const {separator: t, experimentalParseClassName: n} = e,
			r = t.length === 1,
			o = t[0],
			i = t.length,
			s = (l) => {
				const a = [];
				let u = 0,
					f = 0,
					d;
				for (let S = 0; S < l.length; S++) {
					let m = l[S];
					if (u === 0) {
						if (m === o && (r || l.slice(S, S + i) === t)) {
							a.push(l.slice(f, S)), (f = S + i);
							continue;
						}
						if (m === '/') {
							d = S;
							continue;
						}
					}
					m === '[' ? u++ : m === ']' && u--;
				}
				const c = a.length === 0 ? l : l.substring(f),
					g = c.startsWith(Fm),
					w = g ? c.substring(1) : c,
					y = d && d > f ? d - f : void 0;
				return {modifiers: a, hasImportantModifier: g, baseClassName: w, maybePostfixModifierPosition: y};
			};
		return n ? (l) => n({className: l, parseClassName: s}) : s;
	},
	xw = (e) => {
		if (e.length <= 1) return e;
		const t = [];
		let n = [];
		return (
			e.forEach((r) => {
				r[0] === '[' ? (t.push(...n.sort(), r), (n = [])) : n.push(r);
			}),
			t.push(...n.sort()),
			t
		);
	},
	Sw = (e) => ({cache: yw(e.cacheSize), parseClassName: ww(e), ...pw(e)}),
	Ew = /\s+/,
	Cw = (e, t) => {
		const {parseClassName: n, getClassGroupId: r, getConflictingClassGroupIds: o} = t,
			i = [],
			s = e.trim().split(Ew);
		let l = '';
		for (let a = s.length - 1; a >= 0; a -= 1) {
			const u = s[a],
				{modifiers: f, hasImportantModifier: d, baseClassName: c, maybePostfixModifierPosition: g} = n(u);
			let w = !!g,
				y = r(w ? c.substring(0, g) : c);
			if (!y) {
				if (!w) {
					l = u + (l.length > 0 ? ' ' + l : l);
					continue;
				}
				if (((y = r(c)), !y)) {
					l = u + (l.length > 0 ? ' ' + l : l);
					continue;
				}
				w = !1;
			}
			const S = xw(f).join(':'),
				m = d ? S + Fm : S,
				h = m + y;
			if (i.includes(h)) continue;
			i.push(h);
			const x = o(y, w);
			for (let E = 0; E < x.length; ++E) {
				const C = x[E];
				i.push(m + C);
			}
			l = u + (l.length > 0 ? ' ' + l : l);
		}
		return l;
	};
function zm() {
	let e = 0,
		t,
		n,
		r = '';
	for (; e < arguments.length; ) (t = arguments[e++]) && (n = $m(t)) && (r && (r += ' '), (r += n));
	return r;
}
const $m = (e) => {
	if (typeof e == 'string') return e;
	let t,
		n = '';
	for (let r = 0; r < e.length; r++) e[r] && (t = $m(e[r])) && (n && (n += ' '), (n += t));
	return n;
};
function bw(e, ...t) {
	let n,
		r,
		o,
		i = s;
	function s(a) {
		const u = t.reduce((f, d) => d(f), e());
		return (n = Sw(u)), (r = n.cache.get), (o = n.cache.set), (i = l), l(a);
	}
	function l(a) {
		const u = r(a);
		if (u) return u;
		const f = Cw(a, n);
		return o(a, f), f;
	}
	return function () {
		return i(zm.apply(null, arguments));
	};
}
const ne = (e) => {
		const t = (n) => n[e] || [];
		return (t.isThemeGetter = !0), t;
	},
	Bm = /^\[(?:([a-z-]+):)?(.+)\]$/i,
	kw = /^\d+\/\d+$/,
	Tw = new Set(['px', 'full', 'screen']),
	Pw = /^(\d+(\.\d+)?)?(xs|sm|md|lg|xl)$/,
	Nw = /\d+(%|px|r?em|[sdl]?v([hwib]|min|max)|pt|pc|in|cm|mm|cap|ch|ex|r?lh|cq(w|h|i|b|min|max))|\b(calc|min|max|clamp)\(.+\)|^0$/,
	_w = /^(rgba?|hsla?|hwb|(ok)?(lab|lch))\(.+\)$/,
	Rw = /^(inset_)?-?((\d+)?\.?(\d+)[a-z]+|0)_-?((\d+)?\.?(\d+)[a-z]+|0)/,
	Aw = /^(url|image|image-set|cross-fade|element|(repeating-)?(linear|radial|conic)-gradient)\(.+\)$/,
	Rt = (e) => vr(e) || Tw.has(e) || kw.test(e),
	Yt = (e) => Ir(e, 'length', zw),
	vr = (e) => !!e && !Number.isNaN(Number(e)),
	_l = (e) => Ir(e, 'number', vr),
	Kr = (e) => !!e && Number.isInteger(Number(e)),
	Mw = (e) => e.endsWith('%') && vr(e.slice(0, -1)),
	V = (e) => Bm.test(e),
	Qt = (e) => Pw.test(e),
	Ow = new Set(['length', 'size', 'percentage']),
	jw = (e) => Ir(e, Ow, Um),
	Iw = (e) => Ir(e, 'position', Um),
	Dw = new Set(['image', 'url']),
	Lw = (e) => Ir(e, Dw, Bw),
	Fw = (e) => Ir(e, '', $w),
	Yr = () => !0,
	Ir = (e, t, n) => {
		const r = Bm.exec(e);
		return r ? (r[1] ? (typeof t == 'string' ? r[1] === t : t.has(r[1])) : n(r[2])) : !1;
	},
	zw = (e) => Nw.test(e) && !_w.test(e),
	Um = () => !1,
	$w = (e) => Rw.test(e),
	Bw = (e) => Aw.test(e),
	Uw = () => {
		const e = ne('colors'),
			t = ne('spacing'),
			n = ne('blur'),
			r = ne('brightness'),
			o = ne('borderColor'),
			i = ne('borderRadius'),
			s = ne('borderSpacing'),
			l = ne('borderWidth'),
			a = ne('contrast'),
			u = ne('grayscale'),
			f = ne('hueRotate'),
			d = ne('invert'),
			c = ne('gap'),
			g = ne('gradientColorStops'),
			w = ne('gradientColorStopPositions'),
			y = ne('inset'),
			S = ne('margin'),
			m = ne('opacity'),
			h = ne('padding'),
			x = ne('saturate'),
			E = ne('scale'),
			C = ne('sepia'),
			T = ne('skew'),
			k = ne('space'),
			N = ne('translate'),
			D = () => ['auto', 'contain', 'none'],
			M = () => ['auto', 'hidden', 'clip', 'visible', 'scroll'],
			z = () => ['auto', V, t],
			R = () => [V, t],
			$ = () => ['', Rt, Yt],
			U = () => ['auto', vr, V],
			Q = () => ['bottom', 'center', 'left', 'left-bottom', 'left-top', 'right', 'right-bottom', 'right-top', 'top'],
			F = () => ['solid', 'dashed', 'dotted', 'double', 'none'],
			X = () => ['normal', 'multiply', 'screen', 'overlay', 'darken', 'lighten', 'color-dodge', 'color-burn', 'hard-light', 'soft-light', 'difference', 'exclusion', 'hue', 'saturation', 'color', 'luminosity'],
			b = () => ['start', 'end', 'center', 'between', 'around', 'evenly', 'stretch'],
			P = () => ['', '0', V],
			j = () => ['auto', 'avoid', 'all', 'avoid-page', 'page', 'left', 'right', 'column'],
			O = () => [vr, V];
		return {
			cacheSize: 500,
			separator: ':',
			theme: {
				colors: [Yr],
				spacing: [Rt, Yt],
				blur: ['none', '', Qt, V],
				brightness: O(),
				borderColor: [e],
				borderRadius: ['none', '', 'full', Qt, V],
				borderSpacing: R(),
				borderWidth: $(),
				contrast: O(),
				grayscale: P(),
				hueRotate: O(),
				invert: P(),
				gap: R(),
				gradientColorStops: [e],
				gradientColorStopPositions: [Mw, Yt],
				inset: z(),
				margin: z(),
				opacity: O(),
				padding: R(),
				saturate: O(),
				scale: O(),
				sepia: P(),
				skew: O(),
				space: R(),
				translate: R(),
			},
			classGroups: {
				aspect: [{aspect: ['auto', 'square', 'video', V]}],
				container: ['container'],
				columns: [{columns: [Qt]}],
				'break-after': [{'break-after': j()}],
				'break-before': [{'break-before': j()}],
				'break-inside': [{'break-inside': ['auto', 'avoid', 'avoid-page', 'avoid-column']}],
				'box-decoration': [{'box-decoration': ['slice', 'clone']}],
				box: [{box: ['border', 'content']}],
				display: [
					'block',
					'inline-block',
					'inline',
					'flex',
					'inline-flex',
					'table',
					'inline-table',
					'table-caption',
					'table-cell',
					'table-column',
					'table-column-group',
					'table-footer-group',
					'table-header-group',
					'table-row-group',
					'table-row',
					'flow-root',
					'grid',
					'inline-grid',
					'contents',
					'list-item',
					'hidden',
				],
				float: [{float: ['right', 'left', 'none', 'start', 'end']}],
				clear: [{clear: ['left', 'right', 'both', 'none', 'start', 'end']}],
				isolation: ['isolate', 'isolation-auto'],
				'object-fit': [{object: ['contain', 'cover', 'fill', 'none', 'scale-down']}],
				'object-position': [{object: [...Q(), V]}],
				overflow: [{overflow: M()}],
				'overflow-x': [{'overflow-x': M()}],
				'overflow-y': [{'overflow-y': M()}],
				overscroll: [{overscroll: D()}],
				'overscroll-x': [{'overscroll-x': D()}],
				'overscroll-y': [{'overscroll-y': D()}],
				position: ['static', 'fixed', 'absolute', 'relative', 'sticky'],
				inset: [{inset: [y]}],
				'inset-x': [{'inset-x': [y]}],
				'inset-y': [{'inset-y': [y]}],
				start: [{start: [y]}],
				end: [{end: [y]}],
				top: [{top: [y]}],
				right: [{right: [y]}],
				bottom: [{bottom: [y]}],
				left: [{left: [y]}],
				visibility: ['visible', 'invisible', 'collapse'],
				z: [{z: ['auto', Kr, V]}],
				basis: [{basis: z()}],
				'flex-direction': [{flex: ['row', 'row-reverse', 'col', 'col-reverse']}],
				'flex-wrap': [{flex: ['wrap', 'wrap-reverse', 'nowrap']}],
				flex: [{flex: ['1', 'auto', 'initial', 'none', V]}],
				grow: [{grow: P()}],
				shrink: [{shrink: P()}],
				order: [{order: ['first', 'last', 'none', Kr, V]}],
				'grid-cols': [{'grid-cols': [Yr]}],
				'col-start-end': [{col: ['auto', {span: ['full', Kr, V]}, V]}],
				'col-start': [{'col-start': U()}],
				'col-end': [{'col-end': U()}],
				'grid-rows': [{'grid-rows': [Yr]}],
				'row-start-end': [{row: ['auto', {span: [Kr, V]}, V]}],
				'row-start': [{'row-start': U()}],
				'row-end': [{'row-end': U()}],
				'grid-flow': [{'grid-flow': ['row', 'col', 'dense', 'row-dense', 'col-dense']}],
				'auto-cols': [{'auto-cols': ['auto', 'min', 'max', 'fr', V]}],
				'auto-rows': [{'auto-rows': ['auto', 'min', 'max', 'fr', V]}],
				gap: [{gap: [c]}],
				'gap-x': [{'gap-x': [c]}],
				'gap-y': [{'gap-y': [c]}],
				'justify-content': [{justify: ['normal', ...b()]}],
				'justify-items': [{'justify-items': ['start', 'end', 'center', 'stretch']}],
				'justify-self': [{'justify-self': ['auto', 'start', 'end', 'center', 'stretch']}],
				'align-content': [{content: ['normal', ...b(), 'baseline']}],
				'align-items': [{items: ['start', 'end', 'center', 'baseline', 'stretch']}],
				'align-self': [{self: ['auto', 'start', 'end', 'center', 'stretch', 'baseline']}],
				'place-content': [{'place-content': [...b(), 'baseline']}],
				'place-items': [{'place-items': ['start', 'end', 'center', 'baseline', 'stretch']}],
				'place-self': [{'place-self': ['auto', 'start', 'end', 'center', 'stretch']}],
				p: [{p: [h]}],
				px: [{px: [h]}],
				py: [{py: [h]}],
				ps: [{ps: [h]}],
				pe: [{pe: [h]}],
				pt: [{pt: [h]}],
				pr: [{pr: [h]}],
				pb: [{pb: [h]}],
				pl: [{pl: [h]}],
				m: [{m: [S]}],
				mx: [{mx: [S]}],
				my: [{my: [S]}],
				ms: [{ms: [S]}],
				me: [{me: [S]}],
				mt: [{mt: [S]}],
				mr: [{mr: [S]}],
				mb: [{mb: [S]}],
				ml: [{ml: [S]}],
				'space-x': [{'space-x': [k]}],
				'space-x-reverse': ['space-x-reverse'],
				'space-y': [{'space-y': [k]}],
				'space-y-reverse': ['space-y-reverse'],
				w: [{w: ['auto', 'min', 'max', 'fit', 'svw', 'lvw', 'dvw', V, t]}],
				'min-w': [{'min-w': [V, t, 'min', 'max', 'fit']}],
				'max-w': [{'max-w': [V, t, 'none', 'full', 'min', 'max', 'fit', 'prose', {screen: [Qt]}, Qt]}],
				h: [{h: [V, t, 'auto', 'min', 'max', 'fit', 'svh', 'lvh', 'dvh']}],
				'min-h': [{'min-h': [V, t, 'min', 'max', 'fit', 'svh', 'lvh', 'dvh']}],
				'max-h': [{'max-h': [V, t, 'min', 'max', 'fit', 'svh', 'lvh', 'dvh']}],
				size: [{size: [V, t, 'auto', 'min', 'max', 'fit']}],
				'font-size': [{text: ['base', Qt, Yt]}],
				'font-smoothing': ['antialiased', 'subpixel-antialiased'],
				'font-style': ['italic', 'not-italic'],
				'font-weight': [{font: ['thin', 'extralight', 'light', 'normal', 'medium', 'semibold', 'bold', 'extrabold', 'black', _l]}],
				'font-family': [{font: [Yr]}],
				'fvn-normal': ['normal-nums'],
				'fvn-ordinal': ['ordinal'],
				'fvn-slashed-zero': ['slashed-zero'],
				'fvn-figure': ['lining-nums', 'oldstyle-nums'],
				'fvn-spacing': ['proportional-nums', 'tabular-nums'],
				'fvn-fraction': ['diagonal-fractions', 'stacked-fractons'],
				tracking: [{tracking: ['tighter', 'tight', 'normal', 'wide', 'wider', 'widest', V]}],
				'line-clamp': [{'line-clamp': ['none', vr, _l]}],
				leading: [{leading: ['none', 'tight', 'snug', 'normal', 'relaxed', 'loose', Rt, V]}],
				'list-image': [{'list-image': ['none', V]}],
				'list-style-type': [{list: ['none', 'disc', 'decimal', V]}],
				'list-style-position': [{list: ['inside', 'outside']}],
				'placeholder-color': [{placeholder: [e]}],
				'placeholder-opacity': [{'placeholder-opacity': [m]}],
				'text-alignment': [{text: ['left', 'center', 'right', 'justify', 'start', 'end']}],
				'text-color': [{text: [e]}],
				'text-opacity': [{'text-opacity': [m]}],
				'text-decoration': ['underline', 'overline', 'line-through', 'no-underline'],
				'text-decoration-style': [{decoration: [...F(), 'wavy']}],
				'text-decoration-thickness': [{decoration: ['auto', 'from-font', Rt, Yt]}],
				'underline-offset': [{'underline-offset': ['auto', Rt, V]}],
				'text-decoration-color': [{decoration: [e]}],
				'text-transform': ['uppercase', 'lowercase', 'capitalize', 'normal-case'],
				'text-overflow': ['truncate', 'text-ellipsis', 'text-clip'],
				'text-wrap': [{text: ['wrap', 'nowrap', 'balance', 'pretty']}],
				indent: [{indent: R()}],
				'vertical-align': [{align: ['baseline', 'top', 'middle', 'bottom', 'text-top', 'text-bottom', 'sub', 'super', V]}],
				whitespace: [{whitespace: ['normal', 'nowrap', 'pre', 'pre-line', 'pre-wrap', 'break-spaces']}],
				break: [{break: ['normal', 'words', 'all', 'keep']}],
				hyphens: [{hyphens: ['none', 'manual', 'auto']}],
				content: [{content: ['none', V]}],
				'bg-attachment': [{bg: ['fixed', 'local', 'scroll']}],
				'bg-clip': [{'bg-clip': ['border', 'padding', 'content', 'text']}],
				'bg-opacity': [{'bg-opacity': [m]}],
				'bg-origin': [{'bg-origin': ['border', 'padding', 'content']}],
				'bg-position': [{bg: [...Q(), Iw]}],
				'bg-repeat': [{bg: ['no-repeat', {repeat: ['', 'x', 'y', 'round', 'space']}]}],
				'bg-size': [{bg: ['auto', 'cover', 'contain', jw]}],
				'bg-image': [{bg: ['none', {'gradient-to': ['t', 'tr', 'r', 'br', 'b', 'bl', 'l', 'tl']}, Lw]}],
				'bg-color': [{bg: [e]}],
				'gradient-from-pos': [{from: [w]}],
				'gradient-via-pos': [{via: [w]}],
				'gradient-to-pos': [{to: [w]}],
				'gradient-from': [{from: [g]}],
				'gradient-via': [{via: [g]}],
				'gradient-to': [{to: [g]}],
				rounded: [{rounded: [i]}],
				'rounded-s': [{'rounded-s': [i]}],
				'rounded-e': [{'rounded-e': [i]}],
				'rounded-t': [{'rounded-t': [i]}],
				'rounded-r': [{'rounded-r': [i]}],
				'rounded-b': [{'rounded-b': [i]}],
				'rounded-l': [{'rounded-l': [i]}],
				'rounded-ss': [{'rounded-ss': [i]}],
				'rounded-se': [{'rounded-se': [i]}],
				'rounded-ee': [{'rounded-ee': [i]}],
				'rounded-es': [{'rounded-es': [i]}],
				'rounded-tl': [{'rounded-tl': [i]}],
				'rounded-tr': [{'rounded-tr': [i]}],
				'rounded-br': [{'rounded-br': [i]}],
				'rounded-bl': [{'rounded-bl': [i]}],
				'border-w': [{border: [l]}],
				'border-w-x': [{'border-x': [l]}],
				'border-w-y': [{'border-y': [l]}],
				'border-w-s': [{'border-s': [l]}],
				'border-w-e': [{'border-e': [l]}],
				'border-w-t': [{'border-t': [l]}],
				'border-w-r': [{'border-r': [l]}],
				'border-w-b': [{'border-b': [l]}],
				'border-w-l': [{'border-l': [l]}],
				'border-opacity': [{'border-opacity': [m]}],
				'border-style': [{border: [...F(), 'hidden']}],
				'divide-x': [{'divide-x': [l]}],
				'divide-x-reverse': ['divide-x-reverse'],
				'divide-y': [{'divide-y': [l]}],
				'divide-y-reverse': ['divide-y-reverse'],
				'divide-opacity': [{'divide-opacity': [m]}],
				'divide-style': [{divide: F()}],
				'border-color': [{border: [o]}],
				'border-color-x': [{'border-x': [o]}],
				'border-color-y': [{'border-y': [o]}],
				'border-color-s': [{'border-s': [o]}],
				'border-color-e': [{'border-e': [o]}],
				'border-color-t': [{'border-t': [o]}],
				'border-color-r': [{'border-r': [o]}],
				'border-color-b': [{'border-b': [o]}],
				'border-color-l': [{'border-l': [o]}],
				'divide-color': [{divide: [o]}],
				'outline-style': [{outline: ['', ...F()]}],
				'outline-offset': [{'outline-offset': [Rt, V]}],
				'outline-w': [{outline: [Rt, Yt]}],
				'outline-color': [{outline: [e]}],
				'ring-w': [{ring: $()}],
				'ring-w-inset': ['ring-inset'],
				'ring-color': [{ring: [e]}],
				'ring-opacity': [{'ring-opacity': [m]}],
				'ring-offset-w': [{'ring-offset': [Rt, Yt]}],
				'ring-offset-color': [{'ring-offset': [e]}],
				shadow: [{shadow: ['', 'inner', 'none', Qt, Fw]}],
				'shadow-color': [{shadow: [Yr]}],
				opacity: [{opacity: [m]}],
				'mix-blend': [{'mix-blend': [...X(), 'plus-lighter', 'plus-darker']}],
				'bg-blend': [{'bg-blend': X()}],
				filter: [{filter: ['', 'none']}],
				blur: [{blur: [n]}],
				brightness: [{brightness: [r]}],
				contrast: [{contrast: [a]}],
				'drop-shadow': [{'drop-shadow': ['', 'none', Qt, V]}],
				grayscale: [{grayscale: [u]}],
				'hue-rotate': [{'hue-rotate': [f]}],
				invert: [{invert: [d]}],
				saturate: [{saturate: [x]}],
				sepia: [{sepia: [C]}],
				'backdrop-filter': [{'backdrop-filter': ['', 'none']}],
				'backdrop-blur': [{'backdrop-blur': [n]}],
				'backdrop-brightness': [{'backdrop-brightness': [r]}],
				'backdrop-contrast': [{'backdrop-contrast': [a]}],
				'backdrop-grayscale': [{'backdrop-grayscale': [u]}],
				'backdrop-hue-rotate': [{'backdrop-hue-rotate': [f]}],
				'backdrop-invert': [{'backdrop-invert': [d]}],
				'backdrop-opacity': [{'backdrop-opacity': [m]}],
				'backdrop-saturate': [{'backdrop-saturate': [x]}],
				'backdrop-sepia': [{'backdrop-sepia': [C]}],
				'border-collapse': [{border: ['collapse', 'separate']}],
				'border-spacing': [{'border-spacing': [s]}],
				'border-spacing-x': [{'border-spacing-x': [s]}],
				'border-spacing-y': [{'border-spacing-y': [s]}],
				'table-layout': [{table: ['auto', 'fixed']}],
				caption: [{caption: ['top', 'bottom']}],
				transition: [{transition: ['none', 'all', '', 'colors', 'opacity', 'shadow', 'transform', V]}],
				duration: [{duration: O()}],
				ease: [{ease: ['linear', 'in', 'out', 'in-out', V]}],
				delay: [{delay: O()}],
				animate: [{animate: ['none', 'spin', 'ping', 'pulse', 'bounce', V]}],
				transform: [{transform: ['', 'gpu', 'none']}],
				scale: [{scale: [E]}],
				'scale-x': [{'scale-x': [E]}],
				'scale-y': [{'scale-y': [E]}],
				rotate: [{rotate: [Kr, V]}],
				'translate-x': [{'translate-x': [N]}],
				'translate-y': [{'translate-y': [N]}],
				'skew-x': [{'skew-x': [T]}],
				'skew-y': [{'skew-y': [T]}],
				'transform-origin': [{origin: ['center', 'top', 'top-right', 'right', 'bottom-right', 'bottom', 'bottom-left', 'left', 'top-left', V]}],
				accent: [{accent: ['auto', e]}],
				appearance: [{appearance: ['none', 'auto']}],
				cursor: [
					{
						cursor: [
							'auto',
							'default',
							'pointer',
							'wait',
							'text',
							'move',
							'help',
							'not-allowed',
							'none',
							'context-menu',
							'progress',
							'cell',
							'crosshair',
							'vertical-text',
							'alias',
							'copy',
							'no-drop',
							'grab',
							'grabbing',
							'all-scroll',
							'col-resize',
							'row-resize',
							'n-resize',
							'e-resize',
							's-resize',
							'w-resize',
							'ne-resize',
							'nw-resize',
							'se-resize',
							'sw-resize',
							'ew-resize',
							'ns-resize',
							'nesw-resize',
							'nwse-resize',
							'zoom-in',
							'zoom-out',
							V,
						],
					},
				],
				'caret-color': [{caret: [e]}],
				'pointer-events': [{'pointer-events': ['none', 'auto']}],
				resize: [{resize: ['none', 'y', 'x', '']}],
				'scroll-behavior': [{scroll: ['auto', 'smooth']}],
				'scroll-m': [{'scroll-m': R()}],
				'scroll-mx': [{'scroll-mx': R()}],
				'scroll-my': [{'scroll-my': R()}],
				'scroll-ms': [{'scroll-ms': R()}],
				'scroll-me': [{'scroll-me': R()}],
				'scroll-mt': [{'scroll-mt': R()}],
				'scroll-mr': [{'scroll-mr': R()}],
				'scroll-mb': [{'scroll-mb': R()}],
				'scroll-ml': [{'scroll-ml': R()}],
				'scroll-p': [{'scroll-p': R()}],
				'scroll-px': [{'scroll-px': R()}],
				'scroll-py': [{'scroll-py': R()}],
				'scroll-ps': [{'scroll-ps': R()}],
				'scroll-pe': [{'scroll-pe': R()}],
				'scroll-pt': [{'scroll-pt': R()}],
				'scroll-pr': [{'scroll-pr': R()}],
				'scroll-pb': [{'scroll-pb': R()}],
				'scroll-pl': [{'scroll-pl': R()}],
				'snap-align': [{snap: ['start', 'end', 'center', 'align-none']}],
				'snap-stop': [{snap: ['normal', 'always']}],
				'snap-type': [{snap: ['none', 'x', 'y', 'both']}],
				'snap-strictness': [{snap: ['mandatory', 'proximity']}],
				touch: [{touch: ['auto', 'none', 'manipulation']}],
				'touch-x': [{'touch-pan': ['x', 'left', 'right']}],
				'touch-y': [{'touch-pan': ['y', 'up', 'down']}],
				'touch-pz': ['touch-pinch-zoom'],
				select: [{select: ['none', 'text', 'all', 'auto']}],
				'will-change': [{'will-change': ['auto', 'scroll', 'contents', 'transform', V]}],
				fill: [{fill: [e, 'none']}],
				'stroke-w': [{stroke: [Rt, Yt, _l]}],
				stroke: [{stroke: [e, 'none']}],
				sr: ['sr-only', 'not-sr-only'],
				'forced-color-adjust': [{'forced-color-adjust': ['auto', 'none']}],
			},
			conflictingClassGroups: {
				overflow: ['overflow-x', 'overflow-y'],
				overscroll: ['overscroll-x', 'overscroll-y'],
				inset: ['inset-x', 'inset-y', 'start', 'end', 'top', 'right', 'bottom', 'left'],
				'inset-x': ['right', 'left'],
				'inset-y': ['top', 'bottom'],
				flex: ['basis', 'grow', 'shrink'],
				gap: ['gap-x', 'gap-y'],
				p: ['px', 'py', 'ps', 'pe', 'pt', 'pr', 'pb', 'pl'],
				px: ['pr', 'pl'],
				py: ['pt', 'pb'],
				m: ['mx', 'my', 'ms', 'me', 'mt', 'mr', 'mb', 'ml'],
				mx: ['mr', 'ml'],
				my: ['mt', 'mb'],
				size: ['w', 'h'],
				'font-size': ['leading'],
				'fvn-normal': ['fvn-ordinal', 'fvn-slashed-zero', 'fvn-figure', 'fvn-spacing', 'fvn-fraction'],
				'fvn-ordinal': ['fvn-normal'],
				'fvn-slashed-zero': ['fvn-normal'],
				'fvn-figure': ['fvn-normal'],
				'fvn-spacing': ['fvn-normal'],
				'fvn-fraction': ['fvn-normal'],
				'line-clamp': ['display', 'overflow'],
				rounded: ['rounded-s', 'rounded-e', 'rounded-t', 'rounded-r', 'rounded-b', 'rounded-l', 'rounded-ss', 'rounded-se', 'rounded-ee', 'rounded-es', 'rounded-tl', 'rounded-tr', 'rounded-br', 'rounded-bl'],
				'rounded-s': ['rounded-ss', 'rounded-es'],
				'rounded-e': ['rounded-se', 'rounded-ee'],
				'rounded-t': ['rounded-tl', 'rounded-tr'],
				'rounded-r': ['rounded-tr', 'rounded-br'],
				'rounded-b': ['rounded-br', 'rounded-bl'],
				'rounded-l': ['rounded-tl', 'rounded-bl'],
				'border-spacing': ['border-spacing-x', 'border-spacing-y'],
				'border-w': ['border-w-s', 'border-w-e', 'border-w-t', 'border-w-r', 'border-w-b', 'border-w-l'],
				'border-w-x': ['border-w-r', 'border-w-l'],
				'border-w-y': ['border-w-t', 'border-w-b'],
				'border-color': ['border-color-s', 'border-color-e', 'border-color-t', 'border-color-r', 'border-color-b', 'border-color-l'],
				'border-color-x': ['border-color-r', 'border-color-l'],
				'border-color-y': ['border-color-t', 'border-color-b'],
				'scroll-m': ['scroll-mx', 'scroll-my', 'scroll-ms', 'scroll-me', 'scroll-mt', 'scroll-mr', 'scroll-mb', 'scroll-ml'],
				'scroll-mx': ['scroll-mr', 'scroll-ml'],
				'scroll-my': ['scroll-mt', 'scroll-mb'],
				'scroll-p': ['scroll-px', 'scroll-py', 'scroll-ps', 'scroll-pe', 'scroll-pt', 'scroll-pr', 'scroll-pb', 'scroll-pl'],
				'scroll-px': ['scroll-pr', 'scroll-pl'],
				'scroll-py': ['scroll-pt', 'scroll-pb'],
				touch: ['touch-x', 'touch-y', 'touch-pz'],
				'touch-x': ['touch'],
				'touch-y': ['touch'],
				'touch-pz': ['touch'],
			},
			conflictingClassGroupModifiers: {'font-size': ['leading']},
		};
	},
	Ia = bw(Uw);
function he(...e) {
	return Ia(Ku(e));
}
const Hm = p.forwardRef(({className: e, type: t, ...n}, r) =>
	v.jsx('input', {
		type: t,
		className: he(
			'flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50',
			e
		),
		ref: r,
		...n,
	})
);
Hm.displayName = 'Input';
function B(e, t, {checkForDefaultPrevented: n = !0} = {}) {
	return function (o) {
		if ((e == null || e(o), n === !1 || !o.defaultPrevented)) return t == null ? void 0 : t(o);
	};
}
function Hw(e, t) {
	typeof e == 'function' ? e(t) : e != null && (e.current = t);
}
function Ms(...e) {
	return (t) => e.forEach((n) => Hw(n, t));
}
function we(...e) {
	return p.useCallback(Ms(...e), e);
}
function Ww(e, t) {
	const n = p.createContext(t),
		r = (i) => {
			const {children: s, ...l} = i,
				a = p.useMemo(() => l, Object.values(l));
			return v.jsx(n.Provider, {value: a, children: s});
		};
	r.displayName = e + 'Provider';
	function o(i) {
		const s = p.useContext(n);
		if (s) return s;
		if (t !== void 0) return t;
		throw new Error(`\`${i}\` must be used within \`${e}\``);
	}
	return [r, o];
}
function Os(e, t = []) {
	let n = [];
	function r(i, s) {
		const l = p.createContext(s),
			a = n.length;
		n = [...n, s];
		const u = (d) => {
			var m;
			const {scope: c, children: g, ...w} = d,
				y = ((m = c == null ? void 0 : c[e]) == null ? void 0 : m[a]) || l,
				S = p.useMemo(() => w, Object.values(w));
			return v.jsx(y.Provider, {value: S, children: g});
		};
		u.displayName = i + 'Provider';
		function f(d, c) {
			var y;
			const g = ((y = c == null ? void 0 : c[e]) == null ? void 0 : y[a]) || l,
				w = p.useContext(g);
			if (w) return w;
			if (s !== void 0) return s;
			throw new Error(`\`${d}\` must be used within \`${i}\``);
		}
		return [u, f];
	}
	const o = () => {
		const i = n.map((s) => p.createContext(s));
		return function (l) {
			const a = (l == null ? void 0 : l[e]) || i;
			return p.useMemo(() => ({[`__scope${e}`]: {...l, [e]: a}}), [l, a]);
		};
	};
	return (o.scopeName = e), [r, Vw(o, ...t)];
}
function Vw(...e) {
	const t = e[0];
	if (e.length === 1) return t;
	const n = () => {
		const r = e.map((o) => ({useScope: o(), scopeName: o.scopeName}));
		return function (i) {
			const s = r.reduce((l, {useScope: a, scopeName: u}) => {
				const d = a(i)[`__scope${u}`];
				return {...l, ...d};
			}, {});
			return p.useMemo(() => ({[`__scope${t.scopeName}`]: s}), [s]);
		};
	};
	return (n.scopeName = t.scopeName), n;
}
var Dn = p.forwardRef((e, t) => {
	const {children: n, ...r} = e,
		o = p.Children.toArray(n),
		i = o.find(Gw);
	if (i) {
		const s = i.props.children,
			l = o.map((a) => (a === i ? (p.Children.count(s) > 1 ? p.Children.only(null) : p.isValidElement(s) ? s.props.children : null) : a));
		return v.jsx(Da, {...r, ref: t, children: p.isValidElement(s) ? p.cloneElement(s, void 0, l) : null});
	}
	return v.jsx(Da, {...r, ref: t, children: n});
});
Dn.displayName = 'Slot';
var Da = p.forwardRef((e, t) => {
	const {children: n, ...r} = e;
	if (p.isValidElement(n)) {
		const o = Yw(n);
		return p.cloneElement(n, {...Kw(r, n.props), ref: t ? Ms(t, o) : o});
	}
	return p.Children.count(n) > 1 ? p.Children.only(null) : null;
});
Da.displayName = 'SlotClone';
var Wm = ({children: e}) => v.jsx(v.Fragment, {children: e});
function Gw(e) {
	return p.isValidElement(e) && e.type === Wm;
}
function Kw(e, t) {
	const n = {...t};
	for (const r in t) {
		const o = e[r],
			i = t[r];
		/^on[A-Z]/.test(r)
			? o && i
				? (n[r] = (...l) => {
						i(...l), o(...l);
				  })
				: o && (n[r] = o)
			: r === 'style'
			? (n[r] = {...o, ...i})
			: r === 'className' && (n[r] = [o, i].filter(Boolean).join(' '));
	}
	return {...e, ...n};
}
function Yw(e) {
	var r, o;
	let t = (r = Object.getOwnPropertyDescriptor(e.props, 'ref')) == null ? void 0 : r.get,
		n = t && 'isReactWarning' in t && t.isReactWarning;
	return n ? e.ref : ((t = (o = Object.getOwnPropertyDescriptor(e, 'ref')) == null ? void 0 : o.get), (n = t && 'isReactWarning' in t && t.isReactWarning), n ? e.props.ref : e.props.ref || e.ref);
}
var Qw = ['a', 'button', 'div', 'form', 'h2', 'h3', 'img', 'input', 'label', 'li', 'nav', 'ol', 'p', 'span', 'svg', 'ul'],
	pe = Qw.reduce((e, t) => {
		const n = p.forwardRef((r, o) => {
			const {asChild: i, ...s} = r,
				l = i ? Dn : t;
			return typeof window < 'u' && (window[Symbol.for('radix-ui')] = !0), v.jsx(l, {...s, ref: o});
		});
		return (n.displayName = `Primitive.${t}`), {...e, [t]: n};
	}, {});
function Vm(e, t) {
	e && As.flushSync(() => e.dispatchEvent(t));
}
function gt(e) {
	const t = p.useRef(e);
	return (
		p.useEffect(() => {
			t.current = e;
		}),
		p.useMemo(
			() =>
				(...n) => {
					var r;
					return (r = t.current) == null ? void 0 : r.call(t, ...n);
				},
			[]
		)
	);
}
function Xw(e, t = globalThis == null ? void 0 : globalThis.document) {
	const n = gt(e);
	p.useEffect(() => {
		const r = (o) => {
			o.key === 'Escape' && n(o);
		};
		return t.addEventListener('keydown', r, {capture: !0}), () => t.removeEventListener('keydown', r, {capture: !0});
	}, [n, t]);
}
var Zw = 'DismissableLayer',
	La = 'dismissableLayer.update',
	qw = 'dismissableLayer.pointerDownOutside',
	Jw = 'dismissableLayer.focusOutside',
	Ud,
	Gm = p.createContext({layers: new Set(), layersWithOutsidePointerEventsDisabled: new Set(), branches: new Set()}),
	js = p.forwardRef((e, t) => {
		const {disableOutsidePointerEvents: n = !1, onEscapeKeyDown: r, onPointerDownOutside: o, onFocusOutside: i, onInteractOutside: s, onDismiss: l, ...a} = e,
			u = p.useContext(Gm),
			[f, d] = p.useState(null),
			c = (f == null ? void 0 : f.ownerDocument) ?? (globalThis == null ? void 0 : globalThis.document),
			[, g] = p.useState({}),
			w = we(t, (k) => d(k)),
			y = Array.from(u.layers),
			[S] = [...u.layersWithOutsidePointerEventsDisabled].slice(-1),
			m = y.indexOf(S),
			h = f ? y.indexOf(f) : -1,
			x = u.layersWithOutsidePointerEventsDisabled.size > 0,
			E = h >= m,
			C = nx((k) => {
				const N = k.target,
					D = [...u.branches].some((M) => M.contains(N));
				!E || D || (o == null || o(k), s == null || s(k), k.defaultPrevented || l == null || l());
			}, c),
			T = rx((k) => {
				const N = k.target;
				[...u.branches].some((M) => M.contains(N)) || (i == null || i(k), s == null || s(k), k.defaultPrevented || l == null || l());
			}, c);
		return (
			Xw((k) => {
				h === u.layers.size - 1 && (r == null || r(k), !k.defaultPrevented && l && (k.preventDefault(), l()));
			}, c),
			p.useEffect(() => {
				if (f)
					return (
						n && (u.layersWithOutsidePointerEventsDisabled.size === 0 && ((Ud = c.body.style.pointerEvents), (c.body.style.pointerEvents = 'none')), u.layersWithOutsidePointerEventsDisabled.add(f)),
						u.layers.add(f),
						Hd(),
						() => {
							n && u.layersWithOutsidePointerEventsDisabled.size === 1 && (c.body.style.pointerEvents = Ud);
						}
					);
			}, [f, c, n, u]),
			p.useEffect(
				() => () => {
					f && (u.layers.delete(f), u.layersWithOutsidePointerEventsDisabled.delete(f), Hd());
				},
				[f, u]
			),
			p.useEffect(() => {
				const k = () => g({});
				return document.addEventListener(La, k), () => document.removeEventListener(La, k);
			}, []),
			v.jsx(pe.div, {
				...a,
				ref: w,
				style: {pointerEvents: x ? (E ? 'auto' : 'none') : void 0, ...e.style},
				onFocusCapture: B(e.onFocusCapture, T.onFocusCapture),
				onBlurCapture: B(e.onBlurCapture, T.onBlurCapture),
				onPointerDownCapture: B(e.onPointerDownCapture, C.onPointerDownCapture),
			})
		);
	});
js.displayName = Zw;
var ex = 'DismissableLayerBranch',
	tx = p.forwardRef((e, t) => {
		const n = p.useContext(Gm),
			r = p.useRef(null),
			o = we(t, r);
		return (
			p.useEffect(() => {
				const i = r.current;
				if (i)
					return (
						n.branches.add(i),
						() => {
							n.branches.delete(i);
						}
					);
			}, [n.branches]),
			v.jsx(pe.div, {...e, ref: o})
		);
	});
tx.displayName = ex;
function nx(e, t = globalThis == null ? void 0 : globalThis.document) {
	const n = gt(e),
		r = p.useRef(!1),
		o = p.useRef(() => {});
	return (
		p.useEffect(() => {
			const i = (l) => {
					if (l.target && !r.current) {
						let a = function () {
							Km(qw, n, u, {discrete: !0});
						};
						const u = {originalEvent: l};
						l.pointerType === 'touch' ? (t.removeEventListener('click', o.current), (o.current = a), t.addEventListener('click', o.current, {once: !0})) : a();
					} else t.removeEventListener('click', o.current);
					r.current = !1;
				},
				s = window.setTimeout(() => {
					t.addEventListener('pointerdown', i);
				}, 0);
			return () => {
				window.clearTimeout(s), t.removeEventListener('pointerdown', i), t.removeEventListener('click', o.current);
			};
		}, [t, n]),
		{onPointerDownCapture: () => (r.current = !0)}
	);
}
function rx(e, t = globalThis == null ? void 0 : globalThis.document) {
	const n = gt(e),
		r = p.useRef(!1);
	return (
		p.useEffect(() => {
			const o = (i) => {
				i.target && !r.current && Km(Jw, n, {originalEvent: i}, {discrete: !1});
			};
			return t.addEventListener('focusin', o), () => t.removeEventListener('focusin', o);
		}, [t, n]),
		{onFocusCapture: () => (r.current = !0), onBlurCapture: () => (r.current = !1)}
	);
}
function Hd() {
	const e = new CustomEvent(La);
	document.dispatchEvent(e);
}
function Km(e, t, n, {discrete: r}) {
	const o = n.originalEvent.target,
		i = new CustomEvent(e, {bubbles: !1, cancelable: !0, detail: n});
	t && o.addEventListener(e, t, {once: !0}), r ? Vm(o, i) : o.dispatchEvent(i);
}
var Ln = globalThis != null && globalThis.document ? p.useLayoutEffect : () => {},
	ox = Zv.useId || (() => {}),
	ix = 0;
function _n(e) {
	const [t, n] = p.useState(ox());
	return (
		Ln(() => {
			n((r) => r ?? String(ix++));
		}, [e]),
		t ? `radix-${t}` : ''
	);
}
const sx = ['top', 'right', 'bottom', 'left'],
	mn = Math.min,
	Ge = Math.max,
	ss = Math.round,
	pi = Math.floor,
	hn = (e) => ({x: e, y: e}),
	lx = {left: 'right', right: 'left', bottom: 'top', top: 'bottom'},
	ax = {start: 'end', end: 'start'};
function Fa(e, t, n) {
	return Ge(e, mn(t, n));
}
function Bt(e, t) {
	return typeof e == 'function' ? e(t) : e;
}
function Ut(e) {
	return e.split('-')[0];
}
function Dr(e) {
	return e.split('-')[1];
}
function Qu(e) {
	return e === 'x' ? 'y' : 'x';
}
function Xu(e) {
	return e === 'y' ? 'height' : 'width';
}
function gn(e) {
	return ['top', 'bottom'].includes(Ut(e)) ? 'y' : 'x';
}
function Zu(e) {
	return Qu(gn(e));
}
function ux(e, t, n) {
	n === void 0 && (n = !1);
	const r = Dr(e),
		o = Zu(e),
		i = Xu(o);
	let s = o === 'x' ? (r === (n ? 'end' : 'start') ? 'right' : 'left') : r === 'start' ? 'bottom' : 'top';
	return t.reference[i] > t.floating[i] && (s = ls(s)), [s, ls(s)];
}
function cx(e) {
	const t = ls(e);
	return [za(e), t, za(t)];
}
function za(e) {
	return e.replace(/start|end/g, (t) => ax[t]);
}
function dx(e, t, n) {
	const r = ['left', 'right'],
		o = ['right', 'left'],
		i = ['top', 'bottom'],
		s = ['bottom', 'top'];
	switch (e) {
		case 'top':
		case 'bottom':
			return n ? (t ? o : r) : t ? r : o;
		case 'left':
		case 'right':
			return t ? i : s;
		default:
			return [];
	}
}
function fx(e, t, n, r) {
	const o = Dr(e);
	let i = dx(Ut(e), n === 'start', r);
	return o && ((i = i.map((s) => s + '-' + o)), t && (i = i.concat(i.map(za)))), i;
}
function ls(e) {
	return e.replace(/left|right|bottom|top/g, (t) => lx[t]);
}
function px(e) {
	return {top: 0, right: 0, bottom: 0, left: 0, ...e};
}
function Ym(e) {
	return typeof e != 'number' ? px(e) : {top: e, right: e, bottom: e, left: e};
}
function as(e) {
	const {x: t, y: n, width: r, height: o} = e;
	return {width: r, height: o, top: n, left: t, right: t + r, bottom: n + o, x: t, y: n};
}
function Wd(e, t, n) {
	let {reference: r, floating: o} = e;
	const i = gn(t),
		s = Zu(t),
		l = Xu(s),
		a = Ut(t),
		u = i === 'y',
		f = r.x + r.width / 2 - o.width / 2,
		d = r.y + r.height / 2 - o.height / 2,
		c = r[l] / 2 - o[l] / 2;
	let g;
	switch (a) {
		case 'top':
			g = {x: f, y: r.y - o.height};
			break;
		case 'bottom':
			g = {x: f, y: r.y + r.height};
			break;
		case 'right':
			g = {x: r.x + r.width, y: d};
			break;
		case 'left':
			g = {x: r.x - o.width, y: d};
			break;
		default:
			g = {x: r.x, y: r.y};
	}
	switch (Dr(t)) {
		case 'start':
			g[s] -= c * (n && u ? -1 : 1);
			break;
		case 'end':
			g[s] += c * (n && u ? -1 : 1);
			break;
	}
	return g;
}
const mx = async (e, t, n) => {
	const {placement: r = 'bottom', strategy: o = 'absolute', middleware: i = [], platform: s} = n,
		l = i.filter(Boolean),
		a = await (s.isRTL == null ? void 0 : s.isRTL(t));
	let u = await s.getElementRects({reference: e, floating: t, strategy: o}),
		{x: f, y: d} = Wd(u, r, a),
		c = r,
		g = {},
		w = 0;
	for (let y = 0; y < l.length; y++) {
		const {name: S, fn: m} = l[y],
			{x: h, y: x, data: E, reset: C} = await m({x: f, y: d, initialPlacement: r, placement: c, strategy: o, middlewareData: g, rects: u, platform: s, elements: {reference: e, floating: t}});
		(f = h ?? f),
			(d = x ?? d),
			(g = {...g, [S]: {...g[S], ...E}}),
			C && w <= 50 && (w++, typeof C == 'object' && (C.placement && (c = C.placement), C.rects && (u = C.rects === !0 ? await s.getElementRects({reference: e, floating: t, strategy: o}) : C.rects), ({x: f, y: d} = Wd(u, c, a))), (y = -1));
	}
	return {x: f, y: d, placement: c, strategy: o, middlewareData: g};
};
async function Ro(e, t) {
	var n;
	t === void 0 && (t = {});
	const {x: r, y: o, platform: i, rects: s, elements: l, strategy: a} = e,
		{boundary: u = 'clippingAncestors', rootBoundary: f = 'viewport', elementContext: d = 'floating', altBoundary: c = !1, padding: g = 0} = Bt(t, e),
		w = Ym(g),
		S = l[c ? (d === 'floating' ? 'reference' : 'floating') : d],
		m = as(
			await i.getClippingRect({
				element: (n = await (i.isElement == null ? void 0 : i.isElement(S))) == null || n ? S : S.contextElement || (await (i.getDocumentElement == null ? void 0 : i.getDocumentElement(l.floating))),
				boundary: u,
				rootBoundary: f,
				strategy: a,
			})
		),
		h = d === 'floating' ? {x: r, y: o, width: s.floating.width, height: s.floating.height} : s.reference,
		x = await (i.getOffsetParent == null ? void 0 : i.getOffsetParent(l.floating)),
		E = (await (i.isElement == null ? void 0 : i.isElement(x))) ? (await (i.getScale == null ? void 0 : i.getScale(x))) || {x: 1, y: 1} : {x: 1, y: 1},
		C = as(i.convertOffsetParentRelativeRectToViewportRelativeRect ? await i.convertOffsetParentRelativeRectToViewportRelativeRect({elements: l, rect: h, offsetParent: x, strategy: a}) : h);
	return {top: (m.top - C.top + w.top) / E.y, bottom: (C.bottom - m.bottom + w.bottom) / E.y, left: (m.left - C.left + w.left) / E.x, right: (C.right - m.right + w.right) / E.x};
}
const hx = (e) => ({
		name: 'arrow',
		options: e,
		async fn(t) {
			const {x: n, y: r, placement: o, rects: i, platform: s, elements: l, middlewareData: a} = t,
				{element: u, padding: f = 0} = Bt(e, t) || {};
			if (u == null) return {};
			const d = Ym(f),
				c = {x: n, y: r},
				g = Zu(o),
				w = Xu(g),
				y = await s.getDimensions(u),
				S = g === 'y',
				m = S ? 'top' : 'left',
				h = S ? 'bottom' : 'right',
				x = S ? 'clientHeight' : 'clientWidth',
				E = i.reference[w] + i.reference[g] - c[g] - i.floating[w],
				C = c[g] - i.reference[g],
				T = await (s.getOffsetParent == null ? void 0 : s.getOffsetParent(u));
			let k = T ? T[x] : 0;
			(!k || !(await (s.isElement == null ? void 0 : s.isElement(T)))) && (k = l.floating[x] || i.floating[w]);
			const N = E / 2 - C / 2,
				D = k / 2 - y[w] / 2 - 1,
				M = mn(d[m], D),
				z = mn(d[h], D),
				R = M,
				$ = k - y[w] - z,
				U = k / 2 - y[w] / 2 + N,
				Q = Fa(R, U, $),
				F = !a.arrow && Dr(o) != null && U !== Q && i.reference[w] / 2 - (U < R ? M : z) - y[w] / 2 < 0,
				X = F ? (U < R ? U - R : U - $) : 0;
			return {[g]: c[g] + X, data: {[g]: Q, centerOffset: U - Q - X, ...(F && {alignmentOffset: X})}, reset: F};
		},
	}),
	gx = function (e) {
		return (
			e === void 0 && (e = {}),
			{
				name: 'flip',
				options: e,
				async fn(t) {
					var n, r;
					const {placement: o, middlewareData: i, rects: s, initialPlacement: l, platform: a, elements: u} = t,
						{mainAxis: f = !0, crossAxis: d = !0, fallbackPlacements: c, fallbackStrategy: g = 'bestFit', fallbackAxisSideDirection: w = 'none', flipAlignment: y = !0, ...S} = Bt(e, t);
					if ((n = i.arrow) != null && n.alignmentOffset) return {};
					const m = Ut(o),
						h = gn(l),
						x = Ut(l) === l,
						E = await (a.isRTL == null ? void 0 : a.isRTL(u.floating)),
						C = c || (x || !y ? [ls(l)] : cx(l)),
						T = w !== 'none';
					!c && T && C.push(...fx(l, y, w, E));
					const k = [l, ...C],
						N = await Ro(t, S),
						D = [];
					let M = ((r = i.flip) == null ? void 0 : r.overflows) || [];
					if ((f && D.push(N[m]), d)) {
						const U = ux(o, s, E);
						D.push(N[U[0]], N[U[1]]);
					}
					if (((M = [...M, {placement: o, overflows: D}]), !D.every((U) => U <= 0))) {
						var z, R;
						const U = (((z = i.flip) == null ? void 0 : z.index) || 0) + 1,
							Q = k[U];
						if (Q) return {data: {index: U, overflows: M}, reset: {placement: Q}};
						let F = (R = M.filter((X) => X.overflows[0] <= 0).sort((X, b) => X.overflows[1] - b.overflows[1])[0]) == null ? void 0 : R.placement;
						if (!F)
							switch (g) {
								case 'bestFit': {
									var $;
									const X =
										($ = M.filter((b) => {
											if (T) {
												const P = gn(b.placement);
												return P === h || P === 'y';
											}
											return !0;
										})
											.map((b) => [b.placement, b.overflows.filter((P) => P > 0).reduce((P, j) => P + j, 0)])
											.sort((b, P) => b[1] - P[1])[0]) == null
											? void 0
											: $[0];
									X && (F = X);
									break;
								}
								case 'initialPlacement':
									F = l;
									break;
							}
						if (o !== F) return {reset: {placement: F}};
					}
					return {};
				},
			}
		);
	};
function Vd(e, t) {
	return {top: e.top - t.height, right: e.right - t.width, bottom: e.bottom - t.height, left: e.left - t.width};
}
function Gd(e) {
	return sx.some((t) => e[t] >= 0);
}
const vx = function (e) {
	return (
		e === void 0 && (e = {}),
		{
			name: 'hide',
			options: e,
			async fn(t) {
				const {rects: n} = t,
					{strategy: r = 'referenceHidden', ...o} = Bt(e, t);
				switch (r) {
					case 'referenceHidden': {
						const i = await Ro(t, {...o, elementContext: 'reference'}),
							s = Vd(i, n.reference);
						return {data: {referenceHiddenOffsets: s, referenceHidden: Gd(s)}};
					}
					case 'escaped': {
						const i = await Ro(t, {...o, altBoundary: !0}),
							s = Vd(i, n.floating);
						return {data: {escapedOffsets: s, escaped: Gd(s)}};
					}
					default:
						return {};
				}
			},
		}
	);
};
async function yx(e, t) {
	const {placement: n, platform: r, elements: o} = e,
		i = await (r.isRTL == null ? void 0 : r.isRTL(o.floating)),
		s = Ut(n),
		l = Dr(n),
		a = gn(n) === 'y',
		u = ['left', 'top'].includes(s) ? -1 : 1,
		f = i && a ? -1 : 1,
		d = Bt(t, e);
	let {mainAxis: c, crossAxis: g, alignmentAxis: w} = typeof d == 'number' ? {mainAxis: d, crossAxis: 0, alignmentAxis: null} : {mainAxis: d.mainAxis || 0, crossAxis: d.crossAxis || 0, alignmentAxis: d.alignmentAxis};
	return l && typeof w == 'number' && (g = l === 'end' ? w * -1 : w), a ? {x: g * f, y: c * u} : {x: c * u, y: g * f};
}
const wx = function (e) {
		return (
			e === void 0 && (e = 0),
			{
				name: 'offset',
				options: e,
				async fn(t) {
					var n, r;
					const {x: o, y: i, placement: s, middlewareData: l} = t,
						a = await yx(t, e);
					return s === ((n = l.offset) == null ? void 0 : n.placement) && (r = l.arrow) != null && r.alignmentOffset ? {} : {x: o + a.x, y: i + a.y, data: {...a, placement: s}};
				},
			}
		);
	},
	xx = function (e) {
		return (
			e === void 0 && (e = {}),
			{
				name: 'shift',
				options: e,
				async fn(t) {
					const {x: n, y: r, placement: o} = t,
						{
							mainAxis: i = !0,
							crossAxis: s = !1,
							limiter: l = {
								fn: (S) => {
									let {x: m, y: h} = S;
									return {x: m, y: h};
								},
							},
							...a
						} = Bt(e, t),
						u = {x: n, y: r},
						f = await Ro(t, a),
						d = gn(Ut(o)),
						c = Qu(d);
					let g = u[c],
						w = u[d];
					if (i) {
						const S = c === 'y' ? 'top' : 'left',
							m = c === 'y' ? 'bottom' : 'right',
							h = g + f[S],
							x = g - f[m];
						g = Fa(h, g, x);
					}
					if (s) {
						const S = d === 'y' ? 'top' : 'left',
							m = d === 'y' ? 'bottom' : 'right',
							h = w + f[S],
							x = w - f[m];
						w = Fa(h, w, x);
					}
					const y = l.fn({...t, [c]: g, [d]: w});
					return {...y, data: {x: y.x - n, y: y.y - r, enabled: {[c]: i, [d]: s}}};
				},
			}
		);
	},
	Sx = function (e) {
		return (
			e === void 0 && (e = {}),
			{
				options: e,
				fn(t) {
					const {x: n, y: r, placement: o, rects: i, middlewareData: s} = t,
						{offset: l = 0, mainAxis: a = !0, crossAxis: u = !0} = Bt(e, t),
						f = {x: n, y: r},
						d = gn(o),
						c = Qu(d);
					let g = f[c],
						w = f[d];
					const y = Bt(l, t),
						S = typeof y == 'number' ? {mainAxis: y, crossAxis: 0} : {mainAxis: 0, crossAxis: 0, ...y};
					if (a) {
						const x = c === 'y' ? 'height' : 'width',
							E = i.reference[c] - i.floating[x] + S.mainAxis,
							C = i.reference[c] + i.reference[x] - S.mainAxis;
						g < E ? (g = E) : g > C && (g = C);
					}
					if (u) {
						var m, h;
						const x = c === 'y' ? 'width' : 'height',
							E = ['top', 'left'].includes(Ut(o)),
							C = i.reference[d] - i.floating[x] + ((E && ((m = s.offset) == null ? void 0 : m[d])) || 0) + (E ? 0 : S.crossAxis),
							T = i.reference[d] + i.reference[x] + (E ? 0 : ((h = s.offset) == null ? void 0 : h[d]) || 0) - (E ? S.crossAxis : 0);
						w < C ? (w = C) : w > T && (w = T);
					}
					return {[c]: g, [d]: w};
				},
			}
		);
	},
	Ex = function (e) {
		return (
			e === void 0 && (e = {}),
			{
				name: 'size',
				options: e,
				async fn(t) {
					var n, r;
					const {placement: o, rects: i, platform: s, elements: l} = t,
						{apply: a = () => {}, ...u} = Bt(e, t),
						f = await Ro(t, u),
						d = Ut(o),
						c = Dr(o),
						g = gn(o) === 'y',
						{width: w, height: y} = i.floating;
					let S, m;
					d === 'top' || d === 'bottom' ? ((S = d), (m = c === ((await (s.isRTL == null ? void 0 : s.isRTL(l.floating))) ? 'start' : 'end') ? 'left' : 'right')) : ((m = d), (S = c === 'end' ? 'top' : 'bottom'));
					const h = y - f.top - f.bottom,
						x = w - f.left - f.right,
						E = mn(y - f[S], h),
						C = mn(w - f[m], x),
						T = !t.middlewareData.shift;
					let k = E,
						N = C;
					if (((n = t.middlewareData.shift) != null && n.enabled.x && (N = x), (r = t.middlewareData.shift) != null && r.enabled.y && (k = h), T && !c)) {
						const M = Ge(f.left, 0),
							z = Ge(f.right, 0),
							R = Ge(f.top, 0),
							$ = Ge(f.bottom, 0);
						g ? (N = w - 2 * (M !== 0 || z !== 0 ? M + z : Ge(f.left, f.right))) : (k = y - 2 * (R !== 0 || $ !== 0 ? R + $ : Ge(f.top, f.bottom)));
					}
					await a({...t, availableWidth: N, availableHeight: k});
					const D = await s.getDimensions(l.floating);
					return w !== D.width || y !== D.height ? {reset: {rects: !0}} : {};
				},
			}
		);
	};
function Is() {
	return typeof window < 'u';
}
function Lr(e) {
	return Qm(e) ? (e.nodeName || '').toLowerCase() : '#document';
}
function Qe(e) {
	var t;
	return (e == null || (t = e.ownerDocument) == null ? void 0 : t.defaultView) || window;
}
function _t(e) {
	var t;
	return (t = (Qm(e) ? e.ownerDocument : e.document) || window.document) == null ? void 0 : t.documentElement;
}
function Qm(e) {
	return Is() ? e instanceof Node || e instanceof Qe(e).Node : !1;
}
function vt(e) {
	return Is() ? e instanceof Element || e instanceof Qe(e).Element : !1;
}
function Nt(e) {
	return Is() ? e instanceof HTMLElement || e instanceof Qe(e).HTMLElement : !1;
}
function Kd(e) {
	return !Is() || typeof ShadowRoot > 'u' ? !1 : e instanceof ShadowRoot || e instanceof Qe(e).ShadowRoot;
}
function Uo(e) {
	const {overflow: t, overflowX: n, overflowY: r, display: o} = yt(e);
	return /auto|scroll|overlay|hidden|clip/.test(t + r + n) && !['inline', 'contents'].includes(o);
}
function Cx(e) {
	return ['table', 'td', 'th'].includes(Lr(e));
}
function Ds(e) {
	return [':popover-open', ':modal'].some((t) => {
		try {
			return e.matches(t);
		} catch {
			return !1;
		}
	});
}
function qu(e) {
	const t = Ju(),
		n = vt(e) ? yt(e) : e;
	return (
		n.transform !== 'none' ||
		n.perspective !== 'none' ||
		(n.containerType ? n.containerType !== 'normal' : !1) ||
		(!t && (n.backdropFilter ? n.backdropFilter !== 'none' : !1)) ||
		(!t && (n.filter ? n.filter !== 'none' : !1)) ||
		['transform', 'perspective', 'filter'].some((r) => (n.willChange || '').includes(r)) ||
		['paint', 'layout', 'strict', 'content'].some((r) => (n.contain || '').includes(r))
	);
}
function bx(e) {
	let t = vn(e);
	for (; Nt(t) && !Nr(t); ) {
		if (qu(t)) return t;
		if (Ds(t)) return null;
		t = vn(t);
	}
	return null;
}
function Ju() {
	return typeof CSS > 'u' || !CSS.supports ? !1 : CSS.supports('-webkit-backdrop-filter', 'none');
}
function Nr(e) {
	return ['html', 'body', '#document'].includes(Lr(e));
}
function yt(e) {
	return Qe(e).getComputedStyle(e);
}
function Ls(e) {
	return vt(e) ? {scrollLeft: e.scrollLeft, scrollTop: e.scrollTop} : {scrollLeft: e.scrollX, scrollTop: e.scrollY};
}
function vn(e) {
	if (Lr(e) === 'html') return e;
	const t = e.assignedSlot || e.parentNode || (Kd(e) && e.host) || _t(e);
	return Kd(t) ? t.host : t;
}
function Xm(e) {
	const t = vn(e);
	return Nr(t) ? (e.ownerDocument ? e.ownerDocument.body : e.body) : Nt(t) && Uo(t) ? t : Xm(t);
}
function Ao(e, t, n) {
	var r;
	t === void 0 && (t = []), n === void 0 && (n = !0);
	const o = Xm(e),
		i = o === ((r = e.ownerDocument) == null ? void 0 : r.body),
		s = Qe(o);
	if (i) {
		const l = $a(s);
		return t.concat(s, s.visualViewport || [], Uo(o) ? o : [], l && n ? Ao(l) : []);
	}
	return t.concat(o, Ao(o, [], n));
}
function $a(e) {
	return e.parent && Object.getPrototypeOf(e.parent) ? e.frameElement : null;
}
function Zm(e) {
	const t = yt(e);
	let n = parseFloat(t.width) || 0,
		r = parseFloat(t.height) || 0;
	const o = Nt(e),
		i = o ? e.offsetWidth : n,
		s = o ? e.offsetHeight : r,
		l = ss(n) !== i || ss(r) !== s;
	return l && ((n = i), (r = s)), {width: n, height: r, $: l};
}
function ec(e) {
	return vt(e) ? e : e.contextElement;
}
function yr(e) {
	const t = ec(e);
	if (!Nt(t)) return hn(1);
	const n = t.getBoundingClientRect(),
		{width: r, height: o, $: i} = Zm(t);
	let s = (i ? ss(n.width) : n.width) / r,
		l = (i ? ss(n.height) : n.height) / o;
	return (!s || !Number.isFinite(s)) && (s = 1), (!l || !Number.isFinite(l)) && (l = 1), {x: s, y: l};
}
const kx = hn(0);
function qm(e) {
	const t = Qe(e);
	return !Ju() || !t.visualViewport ? kx : {x: t.visualViewport.offsetLeft, y: t.visualViewport.offsetTop};
}
function Tx(e, t, n) {
	return t === void 0 && (t = !1), !n || (t && n !== Qe(e)) ? !1 : t;
}
function Fn(e, t, n, r) {
	t === void 0 && (t = !1), n === void 0 && (n = !1);
	const o = e.getBoundingClientRect(),
		i = ec(e);
	let s = hn(1);
	t && (r ? vt(r) && (s = yr(r)) : (s = yr(e)));
	const l = Tx(i, n, r) ? qm(i) : hn(0);
	let a = (o.left + l.x) / s.x,
		u = (o.top + l.y) / s.y,
		f = o.width / s.x,
		d = o.height / s.y;
	if (i) {
		const c = Qe(i),
			g = r && vt(r) ? Qe(r) : r;
		let w = c,
			y = $a(w);
		for (; y && r && g !== w; ) {
			const S = yr(y),
				m = y.getBoundingClientRect(),
				h = yt(y),
				x = m.left + (y.clientLeft + parseFloat(h.paddingLeft)) * S.x,
				E = m.top + (y.clientTop + parseFloat(h.paddingTop)) * S.y;
			(a *= S.x), (u *= S.y), (f *= S.x), (d *= S.y), (a += x), (u += E), (w = Qe(y)), (y = $a(w));
		}
	}
	return as({width: f, height: d, x: a, y: u});
}
function Px(e) {
	let {elements: t, rect: n, offsetParent: r, strategy: o} = e;
	const i = o === 'fixed',
		s = _t(r),
		l = t ? Ds(t.floating) : !1;
	if (r === s || (l && i)) return n;
	let a = {scrollLeft: 0, scrollTop: 0},
		u = hn(1);
	const f = hn(0),
		d = Nt(r);
	if ((d || (!d && !i)) && ((Lr(r) !== 'body' || Uo(s)) && (a = Ls(r)), Nt(r))) {
		const c = Fn(r);
		(u = yr(r)), (f.x = c.x + r.clientLeft), (f.y = c.y + r.clientTop);
	}
	return {width: n.width * u.x, height: n.height * u.y, x: n.x * u.x - a.scrollLeft * u.x + f.x, y: n.y * u.y - a.scrollTop * u.y + f.y};
}
function Nx(e) {
	return Array.from(e.getClientRects());
}
function Ba(e, t) {
	const n = Ls(e).scrollLeft;
	return t ? t.left + n : Fn(_t(e)).left + n;
}
function _x(e) {
	const t = _t(e),
		n = Ls(e),
		r = e.ownerDocument.body,
		o = Ge(t.scrollWidth, t.clientWidth, r.scrollWidth, r.clientWidth),
		i = Ge(t.scrollHeight, t.clientHeight, r.scrollHeight, r.clientHeight);
	let s = -n.scrollLeft + Ba(e);
	const l = -n.scrollTop;
	return yt(r).direction === 'rtl' && (s += Ge(t.clientWidth, r.clientWidth) - o), {width: o, height: i, x: s, y: l};
}
function Rx(e, t) {
	const n = Qe(e),
		r = _t(e),
		o = n.visualViewport;
	let i = r.clientWidth,
		s = r.clientHeight,
		l = 0,
		a = 0;
	if (o) {
		(i = o.width), (s = o.height);
		const u = Ju();
		(!u || (u && t === 'fixed')) && ((l = o.offsetLeft), (a = o.offsetTop));
	}
	return {width: i, height: s, x: l, y: a};
}
function Ax(e, t) {
	const n = Fn(e, !0, t === 'fixed'),
		r = n.top + e.clientTop,
		o = n.left + e.clientLeft,
		i = Nt(e) ? yr(e) : hn(1),
		s = e.clientWidth * i.x,
		l = e.clientHeight * i.y,
		a = o * i.x,
		u = r * i.y;
	return {width: s, height: l, x: a, y: u};
}
function Yd(e, t, n) {
	let r;
	if (t === 'viewport') r = Rx(e, n);
	else if (t === 'document') r = _x(_t(e));
	else if (vt(t)) r = Ax(t, n);
	else {
		const o = qm(e);
		r = {...t, x: t.x - o.x, y: t.y - o.y};
	}
	return as(r);
}
function Jm(e, t) {
	const n = vn(e);
	return n === t || !vt(n) || Nr(n) ? !1 : yt(n).position === 'fixed' || Jm(n, t);
}
function Mx(e, t) {
	const n = t.get(e);
	if (n) return n;
	let r = Ao(e, [], !1).filter((l) => vt(l) && Lr(l) !== 'body'),
		o = null;
	const i = yt(e).position === 'fixed';
	let s = i ? vn(e) : e;
	for (; vt(s) && !Nr(s); ) {
		const l = yt(s),
			a = qu(s);
		!a && l.position === 'fixed' && (o = null), (i ? !a && !o : (!a && l.position === 'static' && !!o && ['absolute', 'fixed'].includes(o.position)) || (Uo(s) && !a && Jm(e, s))) ? (r = r.filter((f) => f !== s)) : (o = l), (s = vn(s));
	}
	return t.set(e, r), r;
}
function Ox(e) {
	let {element: t, boundary: n, rootBoundary: r, strategy: o} = e;
	const s = [...(n === 'clippingAncestors' ? (Ds(t) ? [] : Mx(t, this._c)) : [].concat(n)), r],
		l = s[0],
		a = s.reduce((u, f) => {
			const d = Yd(t, f, o);
			return (u.top = Ge(d.top, u.top)), (u.right = mn(d.right, u.right)), (u.bottom = mn(d.bottom, u.bottom)), (u.left = Ge(d.left, u.left)), u;
		}, Yd(t, l, o));
	return {width: a.right - a.left, height: a.bottom - a.top, x: a.left, y: a.top};
}
function jx(e) {
	const {width: t, height: n} = Zm(e);
	return {width: t, height: n};
}
function Ix(e, t, n) {
	const r = Nt(t),
		o = _t(t),
		i = n === 'fixed',
		s = Fn(e, !0, i, t);
	let l = {scrollLeft: 0, scrollTop: 0};
	const a = hn(0);
	if (r || (!r && !i))
		if (((Lr(t) !== 'body' || Uo(o)) && (l = Ls(t)), r)) {
			const g = Fn(t, !0, i, t);
			(a.x = g.x + t.clientLeft), (a.y = g.y + t.clientTop);
		} else o && (a.x = Ba(o));
	let u = 0,
		f = 0;
	if (o && !r && !i) {
		const g = o.getBoundingClientRect();
		(f = g.top + l.scrollTop), (u = g.left + l.scrollLeft - Ba(o, g));
	}
	const d = s.left + l.scrollLeft - a.x - u,
		c = s.top + l.scrollTop - a.y - f;
	return {x: d, y: c, width: s.width, height: s.height};
}
function Rl(e) {
	return yt(e).position === 'static';
}
function Qd(e, t) {
	if (!Nt(e) || yt(e).position === 'fixed') return null;
	if (t) return t(e);
	let n = e.offsetParent;
	return _t(e) === n && (n = n.ownerDocument.body), n;
}
function eh(e, t) {
	const n = Qe(e);
	if (Ds(e)) return n;
	if (!Nt(e)) {
		let o = vn(e);
		for (; o && !Nr(o); ) {
			if (vt(o) && !Rl(o)) return o;
			o = vn(o);
		}
		return n;
	}
	let r = Qd(e, t);
	for (; r && Cx(r) && Rl(r); ) r = Qd(r, t);
	return r && Nr(r) && Rl(r) && !qu(r) ? n : r || bx(e) || n;
}
const Dx = async function (e) {
	const t = this.getOffsetParent || eh,
		n = this.getDimensions,
		r = await n(e.floating);
	return {reference: Ix(e.reference, await t(e.floating), e.strategy), floating: {x: 0, y: 0, width: r.width, height: r.height}};
};
function Lx(e) {
	return yt(e).direction === 'rtl';
}
const Fx = {convertOffsetParentRelativeRectToViewportRelativeRect: Px, getDocumentElement: _t, getClippingRect: Ox, getOffsetParent: eh, getElementRects: Dx, getClientRects: Nx, getDimensions: jx, getScale: yr, isElement: vt, isRTL: Lx};
function zx(e, t) {
	let n = null,
		r;
	const o = _t(e);
	function i() {
		var l;
		clearTimeout(r), (l = n) == null || l.disconnect(), (n = null);
	}
	function s(l, a) {
		l === void 0 && (l = !1), a === void 0 && (a = 1), i();
		const {left: u, top: f, width: d, height: c} = e.getBoundingClientRect();
		if ((l || t(), !d || !c)) return;
		const g = pi(f),
			w = pi(o.clientWidth - (u + d)),
			y = pi(o.clientHeight - (f + c)),
			S = pi(u),
			h = {rootMargin: -g + 'px ' + -w + 'px ' + -y + 'px ' + -S + 'px', threshold: Ge(0, mn(1, a)) || 1};
		let x = !0;
		function E(C) {
			const T = C[0].intersectionRatio;
			if (T !== a) {
				if (!x) return s();
				T
					? s(!1, T)
					: (r = setTimeout(() => {
							s(!1, 1e-7);
					  }, 1e3));
			}
			x = !1;
		}
		try {
			n = new IntersectionObserver(E, {...h, root: o.ownerDocument});
		} catch {
			n = new IntersectionObserver(E, h);
		}
		n.observe(e);
	}
	return s(!0), i;
}
function $x(e, t, n, r) {
	r === void 0 && (r = {});
	const {ancestorScroll: o = !0, ancestorResize: i = !0, elementResize: s = typeof ResizeObserver == 'function', layoutShift: l = typeof IntersectionObserver == 'function', animationFrame: a = !1} = r,
		u = ec(e),
		f = o || i ? [...(u ? Ao(u) : []), ...Ao(t)] : [];
	f.forEach((m) => {
		o && m.addEventListener('scroll', n, {passive: !0}), i && m.addEventListener('resize', n);
	});
	const d = u && l ? zx(u, n) : null;
	let c = -1,
		g = null;
	s &&
		((g = new ResizeObserver((m) => {
			let [h] = m;
			h &&
				h.target === u &&
				g &&
				(g.unobserve(t),
				cancelAnimationFrame(c),
				(c = requestAnimationFrame(() => {
					var x;
					(x = g) == null || x.observe(t);
				}))),
				n();
		})),
		u && !a && g.observe(u),
		g.observe(t));
	let w,
		y = a ? Fn(e) : null;
	a && S();
	function S() {
		const m = Fn(e);
		y && (m.x !== y.x || m.y !== y.y || m.width !== y.width || m.height !== y.height) && n(), (y = m), (w = requestAnimationFrame(S));
	}
	return (
		n(),
		() => {
			var m;
			f.forEach((h) => {
				o && h.removeEventListener('scroll', n), i && h.removeEventListener('resize', n);
			}),
				d == null || d(),
				(m = g) == null || m.disconnect(),
				(g = null),
				a && cancelAnimationFrame(w);
		}
	);
}
const Bx = wx,
	Ux = xx,
	Hx = gx,
	Wx = Ex,
	Vx = vx,
	Xd = hx,
	Gx = Sx,
	Kx = (e, t, n) => {
		const r = new Map(),
			o = {platform: Fx, ...n},
			i = {...o.platform, _c: r};
		return mx(e, t, {...o, platform: i});
	};
var Mi = typeof document < 'u' ? p.useLayoutEffect : p.useEffect;
function us(e, t) {
	if (e === t) return !0;
	if (typeof e != typeof t) return !1;
	if (typeof e == 'function' && e.toString() === t.toString()) return !0;
	let n, r, o;
	if (e && t && typeof e == 'object') {
		if (Array.isArray(e)) {
			if (((n = e.length), n !== t.length)) return !1;
			for (r = n; r-- !== 0; ) if (!us(e[r], t[r])) return !1;
			return !0;
		}
		if (((o = Object.keys(e)), (n = o.length), n !== Object.keys(t).length)) return !1;
		for (r = n; r-- !== 0; ) if (!{}.hasOwnProperty.call(t, o[r])) return !1;
		for (r = n; r-- !== 0; ) {
			const i = o[r];
			if (!(i === '_owner' && e.$$typeof) && !us(e[i], t[i])) return !1;
		}
		return !0;
	}
	return e !== e && t !== t;
}
function th(e) {
	return typeof window > 'u' ? 1 : (e.ownerDocument.defaultView || window).devicePixelRatio || 1;
}
function Zd(e, t) {
	const n = th(e);
	return Math.round(t * n) / n;
}
function Al(e) {
	const t = p.useRef(e);
	return (
		Mi(() => {
			t.current = e;
		}),
		t
	);
}
function Yx(e) {
	e === void 0 && (e = {});
	const {placement: t = 'bottom', strategy: n = 'absolute', middleware: r = [], platform: o, elements: {reference: i, floating: s} = {}, transform: l = !0, whileElementsMounted: a, open: u} = e,
		[f, d] = p.useState({x: 0, y: 0, strategy: n, placement: t, middlewareData: {}, isPositioned: !1}),
		[c, g] = p.useState(r);
	us(c, r) || g(r);
	const [w, y] = p.useState(null),
		[S, m] = p.useState(null),
		h = p.useCallback((b) => {
			b !== T.current && ((T.current = b), y(b));
		}, []),
		x = p.useCallback((b) => {
			b !== k.current && ((k.current = b), m(b));
		}, []),
		E = i || w,
		C = s || S,
		T = p.useRef(null),
		k = p.useRef(null),
		N = p.useRef(f),
		D = a != null,
		M = Al(a),
		z = Al(o),
		R = Al(u),
		$ = p.useCallback(() => {
			if (!T.current || !k.current) return;
			const b = {placement: t, strategy: n, middleware: c};
			z.current && (b.platform = z.current),
				Kx(T.current, k.current, b).then((P) => {
					const j = {...P, isPositioned: R.current !== !1};
					U.current &&
						!us(N.current, j) &&
						((N.current = j),
						As.flushSync(() => {
							d(j);
						}));
				});
		}, [c, t, n, z, R]);
	Mi(() => {
		u === !1 && N.current.isPositioned && ((N.current.isPositioned = !1), d((b) => ({...b, isPositioned: !1})));
	}, [u]);
	const U = p.useRef(!1);
	Mi(
		() => (
			(U.current = !0),
			() => {
				U.current = !1;
			}
		),
		[]
	),
		Mi(() => {
			if ((E && (T.current = E), C && (k.current = C), E && C)) {
				if (M.current) return M.current(E, C, $);
				$();
			}
		}, [E, C, $, M, D]);
	const Q = p.useMemo(() => ({reference: T, floating: k, setReference: h, setFloating: x}), [h, x]),
		F = p.useMemo(() => ({reference: E, floating: C}), [E, C]),
		X = p.useMemo(() => {
			const b = {position: n, left: 0, top: 0};
			if (!F.floating) return b;
			const P = Zd(F.floating, f.x),
				j = Zd(F.floating, f.y);
			return l ? {...b, transform: 'translate(' + P + 'px, ' + j + 'px)', ...(th(F.floating) >= 1.5 && {willChange: 'transform'})} : {position: n, left: P, top: j};
		}, [n, l, F.floating, f.x, f.y]);
	return p.useMemo(() => ({...f, update: $, refs: Q, elements: F, floatingStyles: X}), [f, $, Q, F, X]);
}
const Qx = (e) => {
		function t(n) {
			return {}.hasOwnProperty.call(n, 'current');
		}
		return {
			name: 'arrow',
			options: e,
			fn(n) {
				const {element: r, padding: o} = typeof e == 'function' ? e(n) : e;
				return r && t(r) ? (r.current != null ? Xd({element: r.current, padding: o}).fn(n) : {}) : r ? Xd({element: r, padding: o}).fn(n) : {};
			},
		};
	},
	Xx = (e, t) => ({...Bx(e), options: [e, t]}),
	Zx = (e, t) => ({...Ux(e), options: [e, t]}),
	qx = (e, t) => ({...Gx(e), options: [e, t]}),
	Jx = (e, t) => ({...Hx(e), options: [e, t]}),
	e1 = (e, t) => ({...Wx(e), options: [e, t]}),
	t1 = (e, t) => ({...Vx(e), options: [e, t]}),
	n1 = (e, t) => ({...Qx(e), options: [e, t]});
var r1 = 'Arrow',
	nh = p.forwardRef((e, t) => {
		const {children: n, width: r = 10, height: o = 5, ...i} = e;
		return v.jsx(pe.svg, {...i, ref: t, width: r, height: o, viewBox: '0 0 30 10', preserveAspectRatio: 'none', children: e.asChild ? n : v.jsx('polygon', {points: '0,0 30,0 15,10'})});
	});
nh.displayName = r1;
var o1 = nh;
function i1(e, t = []) {
	let n = [];
	function r(i, s) {
		const l = p.createContext(s),
			a = n.length;
		n = [...n, s];
		function u(d) {
			const {scope: c, children: g, ...w} = d,
				y = (c == null ? void 0 : c[e][a]) || l,
				S = p.useMemo(() => w, Object.values(w));
			return v.jsx(y.Provider, {value: S, children: g});
		}
		function f(d, c) {
			const g = (c == null ? void 0 : c[e][a]) || l,
				w = p.useContext(g);
			if (w) return w;
			if (s !== void 0) return s;
			throw new Error(`\`${d}\` must be used within \`${i}\``);
		}
		return (u.displayName = i + 'Provider'), [u, f];
	}
	const o = () => {
		const i = n.map((s) => p.createContext(s));
		return function (l) {
			const a = (l == null ? void 0 : l[e]) || i;
			return p.useMemo(() => ({[`__scope${e}`]: {...l, [e]: a}}), [l, a]);
		};
	};
	return (o.scopeName = e), [r, s1(o, ...t)];
}
function s1(...e) {
	const t = e[0];
	if (e.length === 1) return t;
	const n = () => {
		const r = e.map((o) => ({useScope: o(), scopeName: o.scopeName}));
		return function (i) {
			const s = r.reduce((l, {useScope: a, scopeName: u}) => {
				const d = a(i)[`__scope${u}`];
				return {...l, ...d};
			}, {});
			return p.useMemo(() => ({[`__scope${t.scopeName}`]: s}), [s]);
		};
	};
	return (n.scopeName = t.scopeName), n;
}
function l1(e) {
	const [t, n] = p.useState(void 0);
	return (
		Ln(() => {
			if (e) {
				n({width: e.offsetWidth, height: e.offsetHeight});
				const r = new ResizeObserver((o) => {
					if (!Array.isArray(o) || !o.length) return;
					const i = o[0];
					let s, l;
					if ('borderBoxSize' in i) {
						const a = i.borderBoxSize,
							u = Array.isArray(a) ? a[0] : a;
						(s = u.inlineSize), (l = u.blockSize);
					} else (s = e.offsetWidth), (l = e.offsetHeight);
					n({width: s, height: l});
				});
				return r.observe(e, {box: 'border-box'}), () => r.unobserve(e);
			} else n(void 0);
		}, [e]),
		t
	);
}
var tc = 'Popper',
	[rh, Fs] = i1(tc),
	[a1, oh] = rh(tc),
	ih = (e) => {
		const {__scopePopper: t, children: n} = e,
			[r, o] = p.useState(null);
		return v.jsx(a1, {scope: t, anchor: r, onAnchorChange: o, children: n});
	};
ih.displayName = tc;
var sh = 'PopperAnchor',
	lh = p.forwardRef((e, t) => {
		const {__scopePopper: n, virtualRef: r, ...o} = e,
			i = oh(sh, n),
			s = p.useRef(null),
			l = we(t, s);
		return (
			p.useEffect(() => {
				i.onAnchorChange((r == null ? void 0 : r.current) || s.current);
			}),
			r ? null : v.jsx(pe.div, {...o, ref: l})
		);
	});
lh.displayName = sh;
var nc = 'PopperContent',
	[u1, c1] = rh(nc),
	ah = p.forwardRef((e, t) => {
		var xe, st, Se, lt, Go, Wn;
		const {
				__scopePopper: n,
				side: r = 'bottom',
				sideOffset: o = 0,
				align: i = 'center',
				alignOffset: s = 0,
				arrowPadding: l = 0,
				avoidCollisions: a = !0,
				collisionBoundary: u = [],
				collisionPadding: f = 0,
				sticky: d = 'partial',
				hideWhenDetached: c = !1,
				updatePositionStrategy: g = 'optimized',
				onPlaced: w,
				...y
			} = e,
			S = oh(nc, n),
			[m, h] = p.useState(null),
			x = we(t, (Sn) => h(Sn)),
			[E, C] = p.useState(null),
			T = l1(E),
			k = (T == null ? void 0 : T.width) ?? 0,
			N = (T == null ? void 0 : T.height) ?? 0,
			D = r + (i !== 'center' ? '-' + i : ''),
			M = typeof f == 'number' ? f : {top: 0, right: 0, bottom: 0, left: 0, ...f},
			z = Array.isArray(u) ? u : [u],
			R = z.length > 0,
			$ = {padding: M, boundary: z.filter(f1), altBoundary: R},
			{
				refs: U,
				floatingStyles: Q,
				placement: F,
				isPositioned: X,
				middlewareData: b,
			} = Yx({
				strategy: 'fixed',
				placement: D,
				whileElementsMounted: (...Sn) => $x(...Sn, {animationFrame: g === 'always'}),
				elements: {reference: S.anchor},
				middleware: [
					Xx({mainAxis: o + N, alignmentAxis: s}),
					a && Zx({mainAxis: !0, crossAxis: !1, limiter: d === 'partial' ? qx() : void 0, ...$}),
					a && Jx({...$}),
					e1({
						...$,
						apply: ({elements: Sn, rects: Fr, availableWidth: Ko, availableHeight: Vt}) => {
							const {width: qs, height: Js} = Fr.reference,
								be = Sn.floating.style;
							be.setProperty('--radix-popper-available-width', `${Ko}px`),
								be.setProperty('--radix-popper-available-height', `${Vt}px`),
								be.setProperty('--radix-popper-anchor-width', `${qs}px`),
								be.setProperty('--radix-popper-anchor-height', `${Js}px`);
						},
					}),
					E && n1({element: E, padding: l}),
					p1({arrowWidth: k, arrowHeight: N}),
					c && t1({strategy: 'referenceHidden', ...$}),
				],
			}),
			[P, j] = dh(F),
			O = gt(w);
		Ln(() => {
			X && (O == null || O());
		}, [X, O]);
		const L = (xe = b.arrow) == null ? void 0 : xe.x,
			Y = (st = b.arrow) == null ? void 0 : st.y,
			ie = ((Se = b.arrow) == null ? void 0 : Se.centerOffset) !== 0,
			[Ne, K] = p.useState();
		return (
			Ln(() => {
				m && K(window.getComputedStyle(m).zIndex);
			}, [m]),
			v.jsx('div', {
				ref: U.setFloating,
				'data-radix-popper-content-wrapper': '',
				style: {
					...Q,
					transform: X ? Q.transform : 'translate(0, -200%)',
					minWidth: 'max-content',
					zIndex: Ne,
					'--radix-popper-transform-origin': [(lt = b.transformOrigin) == null ? void 0 : lt.x, (Go = b.transformOrigin) == null ? void 0 : Go.y].join(' '),
					...(((Wn = b.hide) == null ? void 0 : Wn.referenceHidden) && {visibility: 'hidden', pointerEvents: 'none'}),
				},
				dir: e.dir,
				children: v.jsx(u1, {scope: n, placedSide: P, onArrowChange: C, arrowX: L, arrowY: Y, shouldHideArrow: ie, children: v.jsx(pe.div, {'data-side': P, 'data-align': j, ...y, ref: x, style: {...y.style, animation: X ? void 0 : 'none'}})}),
			})
		);
	});
ah.displayName = nc;
var uh = 'PopperArrow',
	d1 = {top: 'bottom', right: 'left', bottom: 'top', left: 'right'},
	ch = p.forwardRef(function (t, n) {
		const {__scopePopper: r, ...o} = t,
			i = c1(uh, r),
			s = d1[i.placedSide];
		return v.jsx('span', {
			ref: i.onArrowChange,
			style: {
				position: 'absolute',
				left: i.arrowX,
				top: i.arrowY,
				[s]: 0,
				transformOrigin: {top: '', right: '0 0', bottom: 'center 0', left: '100% 0'}[i.placedSide],
				transform: {top: 'translateY(100%)', right: 'translateY(50%) rotate(90deg) translateX(-50%)', bottom: 'rotate(180deg)', left: 'translateY(50%) rotate(-90deg) translateX(50%)'}[i.placedSide],
				visibility: i.shouldHideArrow ? 'hidden' : void 0,
			},
			children: v.jsx(o1, {...o, ref: n, style: {...o.style, display: 'block'}}),
		});
	});
ch.displayName = uh;
function f1(e) {
	return e !== null;
}
var p1 = (e) => ({
	name: 'transformOrigin',
	options: e,
	fn(t) {
		var S, m, h;
		const {placement: n, rects: r, middlewareData: o} = t,
			s = ((S = o.arrow) == null ? void 0 : S.centerOffset) !== 0,
			l = s ? 0 : e.arrowWidth,
			a = s ? 0 : e.arrowHeight,
			[u, f] = dh(n),
			d = {start: '0%', center: '50%', end: '100%'}[f],
			c = (((m = o.arrow) == null ? void 0 : m.x) ?? 0) + l / 2,
			g = (((h = o.arrow) == null ? void 0 : h.y) ?? 0) + a / 2;
		let w = '',
			y = '';
		return (
			u === 'bottom'
				? ((w = s ? d : `${c}px`), (y = `${-a}px`))
				: u === 'top'
				? ((w = s ? d : `${c}px`), (y = `${r.floating.height + a}px`))
				: u === 'right'
				? ((w = `${-a}px`), (y = s ? d : `${g}px`))
				: u === 'left' && ((w = `${r.floating.width + a}px`), (y = s ? d : `${g}px`)),
			{data: {x: w, y}}
		);
	},
});
function dh(e) {
	const [t, n = 'center'] = e.split('-');
	return [t, n];
}
var fh = ih,
	ph = lh,
	mh = ah,
	hh = ch,
	m1 = 'Portal',
	rc = p.forwardRef((e, t) => {
		var l;
		const {container: n, ...r} = e,
			[o, i] = p.useState(!1);
		Ln(() => i(!0), []);
		const s = n || (o && ((l = globalThis == null ? void 0 : globalThis.document) == null ? void 0 : l.body));
		return s ? jm.createPortal(v.jsx(pe.div, {...r, ref: t}), s) : null;
	});
rc.displayName = m1;
function h1(e, t) {
	return p.useReducer((n, r) => t[n][r] ?? n, e);
}
var Wt = (e) => {
	const {present: t, children: n} = e,
		r = g1(t),
		o = typeof n == 'function' ? n({present: r.isPresent}) : p.Children.only(n),
		i = we(r.ref, v1(o));
	return typeof n == 'function' || r.isPresent ? p.cloneElement(o, {ref: i}) : null;
};
Wt.displayName = 'Presence';
function g1(e) {
	const [t, n] = p.useState(),
		r = p.useRef({}),
		o = p.useRef(e),
		i = p.useRef('none'),
		s = e ? 'mounted' : 'unmounted',
		[l, a] = h1(s, {mounted: {UNMOUNT: 'unmounted', ANIMATION_OUT: 'unmountSuspended'}, unmountSuspended: {MOUNT: 'mounted', ANIMATION_END: 'unmounted'}, unmounted: {MOUNT: 'mounted'}});
	return (
		p.useEffect(() => {
			const u = mi(r.current);
			i.current = l === 'mounted' ? u : 'none';
		}, [l]),
		Ln(() => {
			const u = r.current,
				f = o.current;
			if (f !== e) {
				const c = i.current,
					g = mi(u);
				e ? a('MOUNT') : g === 'none' || (u == null ? void 0 : u.display) === 'none' ? a('UNMOUNT') : a(f && c !== g ? 'ANIMATION_OUT' : 'UNMOUNT'), (o.current = e);
			}
		}, [e, a]),
		Ln(() => {
			if (t) {
				let u;
				const f = t.ownerDocument.defaultView ?? window,
					d = (g) => {
						const y = mi(r.current).includes(g.animationName);
						if (g.target === t && y && (a('ANIMATION_END'), !o.current)) {
							const S = t.style.animationFillMode;
							(t.style.animationFillMode = 'forwards'),
								(u = f.setTimeout(() => {
									t.style.animationFillMode === 'forwards' && (t.style.animationFillMode = S);
								}));
						}
					},
					c = (g) => {
						g.target === t && (i.current = mi(r.current));
					};
				return (
					t.addEventListener('animationstart', c),
					t.addEventListener('animationcancel', d),
					t.addEventListener('animationend', d),
					() => {
						f.clearTimeout(u), t.removeEventListener('animationstart', c), t.removeEventListener('animationcancel', d), t.removeEventListener('animationend', d);
					}
				);
			} else a('ANIMATION_END');
		}, [t, a]),
		{
			isPresent: ['mounted', 'unmountSuspended'].includes(l),
			ref: p.useCallback((u) => {
				u && (r.current = getComputedStyle(u)), n(u);
			}, []),
		}
	);
}
function mi(e) {
	return (e == null ? void 0 : e.animationName) || 'none';
}
function v1(e) {
	var r, o;
	let t = (r = Object.getOwnPropertyDescriptor(e.props, 'ref')) == null ? void 0 : r.get,
		n = t && 'isReactWarning' in t && t.isReactWarning;
	return n ? e.ref : ((t = (o = Object.getOwnPropertyDescriptor(e, 'ref')) == null ? void 0 : o.get), (n = t && 'isReactWarning' in t && t.isReactWarning), n ? e.props.ref : e.props.ref || e.ref);
}
function zs({prop: e, defaultProp: t, onChange: n = () => {}}) {
	const [r, o] = y1({defaultProp: t, onChange: n}),
		i = e !== void 0,
		s = i ? e : r,
		l = gt(n),
		a = p.useCallback(
			(u) => {
				if (i) {
					const d = typeof u == 'function' ? u(e) : u;
					d !== e && l(d);
				} else o(u);
			},
			[i, e, o, l]
		);
	return [s, a];
}
function y1({defaultProp: e, onChange: t}) {
	const n = p.useState(e),
		[r] = n,
		o = p.useRef(r),
		i = gt(t);
	return (
		p.useEffect(() => {
			o.current !== r && (i(r), (o.current = r));
		}, [r, o, i]),
		n
	);
}
var w1 = 'VisuallyHidden',
	gh = p.forwardRef((e, t) => v.jsx(pe.span, {...e, ref: t, style: {position: 'absolute', border: 0, width: 1, height: 1, padding: 0, margin: -1, overflow: 'hidden', clip: 'rect(0, 0, 0, 0)', whiteSpace: 'nowrap', wordWrap: 'normal', ...e.style}}));
gh.displayName = w1;
var x1 = gh,
	[$s, Sk] = Os('Tooltip', [Fs]),
	Bs = Fs(),
	vh = 'TooltipProvider',
	S1 = 700,
	Ua = 'tooltip.open',
	[E1, oc] = $s(vh),
	yh = (e) => {
		const {__scopeTooltip: t, delayDuration: n = S1, skipDelayDuration: r = 300, disableHoverableContent: o = !1, children: i} = e,
			[s, l] = p.useState(!0),
			a = p.useRef(!1),
			u = p.useRef(0);
		return (
			p.useEffect(() => {
				const f = u.current;
				return () => window.clearTimeout(f);
			}, []),
			v.jsx(E1, {
				scope: t,
				isOpenDelayed: s,
				delayDuration: n,
				onOpen: p.useCallback(() => {
					window.clearTimeout(u.current), l(!1);
				}, []),
				onClose: p.useCallback(() => {
					window.clearTimeout(u.current), (u.current = window.setTimeout(() => l(!0), r));
				}, [r]),
				isPointerInTransitRef: a,
				onPointerInTransitChange: p.useCallback((f) => {
					a.current = f;
				}, []),
				disableHoverableContent: o,
				children: i,
			})
		);
	};
yh.displayName = vh;
var Us = 'Tooltip',
	[C1, Hs] = $s(Us),
	wh = (e) => {
		const {__scopeTooltip: t, children: n, open: r, defaultOpen: o = !1, onOpenChange: i, disableHoverableContent: s, delayDuration: l} = e,
			a = oc(Us, e.__scopeTooltip),
			u = Bs(t),
			[f, d] = p.useState(null),
			c = _n(),
			g = p.useRef(0),
			w = s ?? a.disableHoverableContent,
			y = l ?? a.delayDuration,
			S = p.useRef(!1),
			[m = !1, h] = zs({
				prop: r,
				defaultProp: o,
				onChange: (k) => {
					k ? (a.onOpen(), document.dispatchEvent(new CustomEvent(Ua))) : a.onClose(), i == null || i(k);
				},
			}),
			x = p.useMemo(() => (m ? (S.current ? 'delayed-open' : 'instant-open') : 'closed'), [m]),
			E = p.useCallback(() => {
				window.clearTimeout(g.current), (S.current = !1), h(!0);
			}, [h]),
			C = p.useCallback(() => {
				window.clearTimeout(g.current), h(!1);
			}, [h]),
			T = p.useCallback(() => {
				window.clearTimeout(g.current),
					(g.current = window.setTimeout(() => {
						(S.current = !0), h(!0);
					}, y));
			}, [y, h]);
		return (
			p.useEffect(() => () => window.clearTimeout(g.current), []),
			v.jsx(fh, {
				...u,
				children: v.jsx(C1, {
					scope: t,
					contentId: c,
					open: m,
					stateAttribute: x,
					trigger: f,
					onTriggerChange: d,
					onTriggerEnter: p.useCallback(() => {
						a.isOpenDelayed ? T() : E();
					}, [a.isOpenDelayed, T, E]),
					onTriggerLeave: p.useCallback(() => {
						w ? C() : window.clearTimeout(g.current);
					}, [C, w]),
					onOpen: E,
					onClose: C,
					disableHoverableContent: w,
					children: n,
				}),
			})
		);
	};
wh.displayName = Us;
var Ha = 'TooltipTrigger',
	xh = p.forwardRef((e, t) => {
		const {__scopeTooltip: n, ...r} = e,
			o = Hs(Ha, n),
			i = oc(Ha, n),
			s = Bs(n),
			l = p.useRef(null),
			a = we(t, l, o.onTriggerChange),
			u = p.useRef(!1),
			f = p.useRef(!1),
			d = p.useCallback(() => (u.current = !1), []);
		return (
			p.useEffect(() => () => document.removeEventListener('pointerup', d), [d]),
			v.jsx(ph, {
				asChild: !0,
				...s,
				children: v.jsx(pe.button, {
					'aria-describedby': o.open ? o.contentId : void 0,
					'data-state': o.stateAttribute,
					...r,
					ref: a,
					onPointerMove: B(e.onPointerMove, (c) => {
						c.pointerType !== 'touch' && !f.current && !i.isPointerInTransitRef.current && (o.onTriggerEnter(), (f.current = !0));
					}),
					onPointerLeave: B(e.onPointerLeave, () => {
						o.onTriggerLeave(), (f.current = !1);
					}),
					onPointerDown: B(e.onPointerDown, () => {
						(u.current = !0), document.addEventListener('pointerup', d, {once: !0});
					}),
					onFocus: B(e.onFocus, () => {
						u.current || o.onOpen();
					}),
					onBlur: B(e.onBlur, o.onClose),
					onClick: B(e.onClick, o.onClose),
				}),
			})
		);
	});
xh.displayName = Ha;
var b1 = 'TooltipPortal',
	[Ek, k1] = $s(b1, {forceMount: void 0}),
	_r = 'TooltipContent',
	Sh = p.forwardRef((e, t) => {
		const n = k1(_r, e.__scopeTooltip),
			{forceMount: r = n.forceMount, side: o = 'top', ...i} = e,
			s = Hs(_r, e.__scopeTooltip);
		return v.jsx(Wt, {present: r || s.open, children: s.disableHoverableContent ? v.jsx(Eh, {side: o, ...i, ref: t}) : v.jsx(T1, {side: o, ...i, ref: t})});
	}),
	T1 = p.forwardRef((e, t) => {
		const n = Hs(_r, e.__scopeTooltip),
			r = oc(_r, e.__scopeTooltip),
			o = p.useRef(null),
			i = we(t, o),
			[s, l] = p.useState(null),
			{trigger: a, onClose: u} = n,
			f = o.current,
			{onPointerInTransitChange: d} = r,
			c = p.useCallback(() => {
				l(null), d(!1);
			}, [d]),
			g = p.useCallback(
				(w, y) => {
					const S = w.currentTarget,
						m = {x: w.clientX, y: w.clientY},
						h = R1(m, S.getBoundingClientRect()),
						x = A1(m, h),
						E = M1(y.getBoundingClientRect()),
						C = j1([...x, ...E]);
					l(C), d(!0);
				},
				[d]
			);
		return (
			p.useEffect(() => () => c(), [c]),
			p.useEffect(() => {
				if (a && f) {
					const w = (S) => g(S, f),
						y = (S) => g(S, a);
					return (
						a.addEventListener('pointerleave', w),
						f.addEventListener('pointerleave', y),
						() => {
							a.removeEventListener('pointerleave', w), f.removeEventListener('pointerleave', y);
						}
					);
				}
			}, [a, f, g, c]),
			p.useEffect(() => {
				if (s) {
					const w = (y) => {
						const S = y.target,
							m = {x: y.clientX, y: y.clientY},
							h = (a == null ? void 0 : a.contains(S)) || (f == null ? void 0 : f.contains(S)),
							x = !O1(m, s);
						h ? c() : x && (c(), u());
					};
					return document.addEventListener('pointermove', w), () => document.removeEventListener('pointermove', w);
				}
			}, [a, f, s, u, c]),
			v.jsx(Eh, {...e, ref: i})
		);
	}),
	[P1, N1] = $s(Us, {isInside: !1}),
	Eh = p.forwardRef((e, t) => {
		const {__scopeTooltip: n, children: r, 'aria-label': o, onEscapeKeyDown: i, onPointerDownOutside: s, ...l} = e,
			a = Hs(_r, n),
			u = Bs(n),
			{onClose: f} = a;
		return (
			p.useEffect(() => (document.addEventListener(Ua, f), () => document.removeEventListener(Ua, f)), [f]),
			p.useEffect(() => {
				if (a.trigger) {
					const d = (c) => {
						const g = c.target;
						g != null && g.contains(a.trigger) && f();
					};
					return window.addEventListener('scroll', d, {capture: !0}), () => window.removeEventListener('scroll', d, {capture: !0});
				}
			}, [a.trigger, f]),
			v.jsx(js, {
				asChild: !0,
				disableOutsidePointerEvents: !1,
				onEscapeKeyDown: i,
				onPointerDownOutside: s,
				onFocusOutside: (d) => d.preventDefault(),
				onDismiss: f,
				children: v.jsxs(mh, {
					'data-state': a.stateAttribute,
					...u,
					...l,
					ref: t,
					style: {
						...l.style,
						'--radix-tooltip-content-transform-origin': 'var(--radix-popper-transform-origin)',
						'--radix-tooltip-content-available-width': 'var(--radix-popper-available-width)',
						'--radix-tooltip-content-available-height': 'var(--radix-popper-available-height)',
						'--radix-tooltip-trigger-width': 'var(--radix-popper-anchor-width)',
						'--radix-tooltip-trigger-height': 'var(--radix-popper-anchor-height)',
					},
					children: [v.jsx(Wm, {children: r}), v.jsx(P1, {scope: n, isInside: !0, children: v.jsx(x1, {id: a.contentId, role: 'tooltip', children: o || r})})],
				}),
			})
		);
	});
Sh.displayName = _r;
var Ch = 'TooltipArrow',
	_1 = p.forwardRef((e, t) => {
		const {__scopeTooltip: n, ...r} = e,
			o = Bs(n);
		return N1(Ch, n).isInside ? null : v.jsx(hh, {...o, ...r, ref: t});
	});
_1.displayName = Ch;
function R1(e, t) {
	const n = Math.abs(t.top - e.y),
		r = Math.abs(t.bottom - e.y),
		o = Math.abs(t.right - e.x),
		i = Math.abs(t.left - e.x);
	switch (Math.min(n, r, o, i)) {
		case i:
			return 'left';
		case o:
			return 'right';
		case n:
			return 'top';
		case r:
			return 'bottom';
		default:
			throw new Error('unreachable');
	}
}
function A1(e, t, n = 5) {
	const r = [];
	switch (t) {
		case 'top':
			r.push({x: e.x - n, y: e.y + n}, {x: e.x + n, y: e.y + n});
			break;
		case 'bottom':
			r.push({x: e.x - n, y: e.y - n}, {x: e.x + n, y: e.y - n});
			break;
		case 'left':
			r.push({x: e.x + n, y: e.y - n}, {x: e.x + n, y: e.y + n});
			break;
		case 'right':
			r.push({x: e.x - n, y: e.y - n}, {x: e.x - n, y: e.y + n});
			break;
	}
	return r;
}
function M1(e) {
	const {top: t, right: n, bottom: r, left: o} = e;
	return [
		{x: o, y: t},
		{x: n, y: t},
		{x: n, y: r},
		{x: o, y: r},
	];
}
function O1(e, t) {
	const {x: n, y: r} = e;
	let o = !1;
	for (let i = 0, s = t.length - 1; i < t.length; s = i++) {
		const l = t[i].x,
			a = t[i].y,
			u = t[s].x,
			f = t[s].y;
		a > r != f > r && n < ((u - l) * (r - a)) / (f - a) + l && (o = !o);
	}
	return o;
}
function j1(e) {
	const t = e.slice();
	return t.sort((n, r) => (n.x < r.x ? -1 : n.x > r.x ? 1 : n.y < r.y ? -1 : n.y > r.y ? 1 : 0)), I1(t);
}
function I1(e) {
	if (e.length <= 1) return e.slice();
	const t = [];
	for (let r = 0; r < e.length; r++) {
		const o = e[r];
		for (; t.length >= 2; ) {
			const i = t[t.length - 1],
				s = t[t.length - 2];
			if ((i.x - s.x) * (o.y - s.y) >= (i.y - s.y) * (o.x - s.x)) t.pop();
			else break;
		}
		t.push(o);
	}
	t.pop();
	const n = [];
	for (let r = e.length - 1; r >= 0; r--) {
		const o = e[r];
		for (; n.length >= 2; ) {
			const i = n[n.length - 1],
				s = n[n.length - 2];
			if ((i.x - s.x) * (o.y - s.y) >= (i.y - s.y) * (o.x - s.x)) n.pop();
			else break;
		}
		n.push(o);
	}
	return n.pop(), t.length === 1 && n.length === 1 && t[0].x === n[0].x && t[0].y === n[0].y ? t : t.concat(n);
}
var D1 = yh,
	L1 = wh,
	F1 = xh,
	bh = Sh;
const z1 = D1,
	$1 = L1,
	B1 = F1,
	kh = p.forwardRef(({className: e, sideOffset: t = 4, ...n}, r) =>
		v.jsx(bh, {
			ref: r,
			sideOffset: t,
			className: he(
				'z-50 overflow-hidden rounded-md border bg-popover px-3 py-1.5 text-sm text-popover-foreground shadow-md animate-in fade-in-0 zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2',
				e
			),
			...n,
		})
	);
kh.displayName = bh.displayName;
function cs({children: e, value: t, style: n = {}, side: r = 'bottom', delayDuration: o = 0}) {
	return v.jsx(z1, {
		children: v.jsxs($1, {
			delayDuration: o,
			children: [
				v.jsx(B1, {asChild: !0, children: e}),
				v.jsx(kh, {
					side: r,
					style: {boxShadow: 'none', ...n},
					className: 'left-[34px] h-[32px] text-[13px] font-normal font-inter rounded-[20px] border border-[#EBECF0] border-solid bg-white p-[5px_16px_7px_16px]',
					children: v.jsx('p', {children: t}),
				}),
			],
		}),
	});
}
function Th(e) {
	var t,
		n,
		r = '';
	if (typeof e == 'string' || typeof e == 'number') r += e;
	else if (typeof e == 'object')
		if (Array.isArray(e)) for (t = 0; t < e.length; t++) e[t] && (n = Th(e[t])) && (r && (r += ' '), (r += n));
		else for (t in e) e[t] && (r && (r += ' '), (r += t));
	return r;
}
function U1() {
	for (var e, t, n = 0, r = ''; n < arguments.length; ) (e = arguments[n++]) && (t = Th(e)) && (r && (r += ' '), (r += t));
	return r;
}
const qd = (e) => (typeof e == 'boolean' ? ''.concat(e) : e === 0 ? '0' : e),
	Jd = U1,
	Ph = (e, t) => (n) => {
		var r;
		if ((t == null ? void 0 : t.variants) == null) return Jd(e, n == null ? void 0 : n.class, n == null ? void 0 : n.className);
		const {variants: o, defaultVariants: i} = t,
			s = Object.keys(o).map((u) => {
				const f = n == null ? void 0 : n[u],
					d = i == null ? void 0 : i[u];
				if (f === null) return null;
				const c = qd(f) || qd(d);
				return o[u][c];
			}),
			l =
				n &&
				Object.entries(n).reduce((u, f) => {
					let [d, c] = f;
					return c === void 0 || (u[d] = c), u;
				}, {}),
			a =
				t == null || (r = t.compoundVariants) === null || r === void 0
					? void 0
					: r.reduce((u, f) => {
							let {class: d, className: c, ...g} = f;
							return Object.entries(g).every((w) => {
								let [y, S] = w;
								return Array.isArray(S) ? S.includes({...i, ...l}[y]) : {...i, ...l}[y] === S;
							})
								? [...u, d, c]
								: u;
					  }, []);
		return Jd(e, s, a, n == null ? void 0 : n.class, n == null ? void 0 : n.className);
	},
	H1 = Ph(
		'inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50',
		{
			variants: {
				variant: {
					default: 'bg-primary text-primary-foreground hover:bg-primary/90',
					destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
					outline: 'border border-input bg-background hover:bg-accent hover:text-accent-foreground',
					secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
					ghost: 'hover:bg-accent hover:text-accent-foreground',
					link: 'text-primary underline-offset-4 hover:underline',
				},
				size: {default: 'h-10 px-4 py-2', sm: 'h-9 rounded-md px-3', lg: 'h-11 rounded-md px-8', icon: 'h-10 w-10'},
			},
			defaultVariants: {variant: 'default', size: 'default'},
		}
	),
	Mo = p.forwardRef(({className: e, variant: t, size: n, asChild: r = !1, ...o}, i) => {
		const s = r ? Dn : 'button';
		return v.jsx(s, {className: he(H1({variant: t, size: n, className: e})), ref: i, ...o});
	});
Mo.displayName = 'Button';
const ic = '',
	sc = async (e, t = {}) => {
		try {
			const n = await fetch(e, t);
			if (!n.ok) throw new Error(`HTTP error! Status: ${n.status}`);
			return t.method === 'PATCH' || t.method === 'DELETE' ? void 0 : await n.json();
		} catch (n) {
			throw (console.error('Fetch error:', n), n);
		}
	},
	Ck = async (e, t) => sc(`${ic}/${e}`, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(t)}),
	W1 = async (e, t) => sc(`${ic}/${e}`, {method: 'PATCH', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(t)}),
	V1 = async (e) => sc(`${ic}/${e}`, {method: 'DELETE'});
var G1 = (e) => {
		switch (e) {
			case 'success':
				return Q1;
			case 'info':
				return Z1;
			case 'warning':
				return X1;
			case 'error':
				return q1;
			default:
				return null;
		}
	},
	K1 = Array(12).fill(0),
	Y1 = ({visible: e}) =>
		A.createElement(
			'div',
			{className: 'sonner-loading-wrapper', 'data-visible': e},
			A.createElement(
				'div',
				{className: 'sonner-spinner'},
				K1.map((t, n) => A.createElement('div', {className: 'sonner-loading-bar', key: `spinner-bar-${n}`}))
			)
		),
	Q1 = A.createElement(
		'svg',
		{xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 20 20', fill: 'currentColor', height: '20', width: '20'},
		A.createElement('path', {fillRule: 'evenodd', d: 'M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z', clipRule: 'evenodd'})
	),
	X1 = A.createElement(
		'svg',
		{xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 24 24', fill: 'currentColor', height: '20', width: '20'},
		A.createElement('path', {
			fillRule: 'evenodd',
			d: 'M9.401 3.003c1.155-2 4.043-2 5.197 0l7.355 12.748c1.154 2-.29 4.5-2.599 4.5H4.645c-2.309 0-3.752-2.5-2.598-4.5L9.4 3.003zM12 8.25a.75.75 0 01.75.75v3.75a.75.75 0 01-1.5 0V9a.75.75 0 01.75-.75zm0 8.25a.75.75 0 100-1.5.75.75 0 000 1.5z',
			clipRule: 'evenodd',
		})
	),
	Z1 = A.createElement(
		'svg',
		{xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 20 20', fill: 'currentColor', height: '20', width: '20'},
		A.createElement('path', {
			fillRule: 'evenodd',
			d: 'M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z',
			clipRule: 'evenodd',
		})
	),
	q1 = A.createElement(
		'svg',
		{xmlns: 'http://www.w3.org/2000/svg', viewBox: '0 0 20 20', fill: 'currentColor', height: '20', width: '20'},
		A.createElement('path', {fillRule: 'evenodd', d: 'M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-5a.75.75 0 01.75.75v4.5a.75.75 0 01-1.5 0v-4.5A.75.75 0 0110 5zm0 10a1 1 0 100-2 1 1 0 000 2z', clipRule: 'evenodd'})
	),
	J1 = () => {
		let [e, t] = A.useState(document.hidden);
		return (
			A.useEffect(() => {
				let n = () => {
					t(document.hidden);
				};
				return document.addEventListener('visibilitychange', n), () => window.removeEventListener('visibilitychange', n);
			}, []),
			e
		);
	},
	Wa = 1,
	eS = class {
		constructor() {
			(this.subscribe = (e) => (
				this.subscribers.push(e),
				() => {
					let t = this.subscribers.indexOf(e);
					this.subscribers.splice(t, 1);
				}
			)),
				(this.publish = (e) => {
					this.subscribers.forEach((t) => t(e));
				}),
				(this.addToast = (e) => {
					this.publish(e), (this.toasts = [...this.toasts, e]);
				}),
				(this.create = (e) => {
					var t;
					let {message: n, ...r} = e,
						o = typeof (e == null ? void 0 : e.id) == 'number' || ((t = e.id) == null ? void 0 : t.length) > 0 ? e.id : Wa++,
						i = this.toasts.find((l) => l.id === o),
						s = e.dismissible === void 0 ? !0 : e.dismissible;
					return i ? (this.toasts = this.toasts.map((l) => (l.id === o ? (this.publish({...l, ...e, id: o, title: n}), {...l, ...e, id: o, dismissible: s, title: n}) : l))) : this.addToast({title: n, ...r, dismissible: s, id: o}), o;
				}),
				(this.dismiss = (e) => (
					e ||
						this.toasts.forEach((t) => {
							this.subscribers.forEach((n) => n({id: t.id, dismiss: !0}));
						}),
					this.subscribers.forEach((t) => t({id: e, dismiss: !0})),
					e
				)),
				(this.message = (e, t) => this.create({...t, message: e})),
				(this.error = (e, t) => this.create({...t, message: e, type: 'error'})),
				(this.success = (e, t) => this.create({...t, type: 'success', message: e})),
				(this.info = (e, t) => this.create({...t, type: 'info', message: e})),
				(this.warning = (e, t) => this.create({...t, type: 'warning', message: e})),
				(this.loading = (e, t) => this.create({...t, type: 'loading', message: e})),
				(this.promise = (e, t) => {
					if (!t) return;
					let n;
					t.loading !== void 0 && (n = this.create({...t, promise: e, type: 'loading', message: t.loading, description: typeof t.description != 'function' ? t.description : void 0}));
					let r = e instanceof Promise ? e : e(),
						o = n !== void 0;
					return (
						r
							.then(async (i) => {
								if (nS(i) && !i.ok) {
									o = !1;
									let s = typeof t.error == 'function' ? await t.error(`HTTP error! status: ${i.status}`) : t.error,
										l = typeof t.description == 'function' ? await t.description(`HTTP error! status: ${i.status}`) : t.description;
									this.create({id: n, type: 'error', message: s, description: l});
								} else if (t.success !== void 0) {
									o = !1;
									let s = typeof t.success == 'function' ? await t.success(i) : t.success,
										l = typeof t.description == 'function' ? await t.description(i) : t.description;
									this.create({id: n, type: 'success', message: s, description: l});
								}
							})
							.catch(async (i) => {
								if (t.error !== void 0) {
									o = !1;
									let s = typeof t.error == 'function' ? await t.error(i) : t.error,
										l = typeof t.description == 'function' ? await t.description(i) : t.description;
									this.create({id: n, type: 'error', message: s, description: l});
								}
							})
							.finally(() => {
								var i;
								o && (this.dismiss(n), (n = void 0)), (i = t.finally) == null || i.call(t);
							}),
						n
					);
				}),
				(this.custom = (e, t) => {
					let n = (t == null ? void 0 : t.id) || Wa++;
					return this.create({jsx: e(n), id: n, ...t}), n;
				}),
				(this.subscribers = []),
				(this.toasts = []);
		}
	},
	We = new eS(),
	tS = (e, t) => {
		let n = (t == null ? void 0 : t.id) || Wa++;
		return We.addToast({title: e, ...t, id: n}), n;
	},
	nS = (e) => e && typeof e == 'object' && 'ok' in e && typeof e.ok == 'boolean' && 'status' in e && typeof e.status == 'number',
	rS = tS,
	oS = () => We.toasts,
	Qr = Object.assign(rS, {success: We.success, info: We.info, warning: We.warning, error: We.error, custom: We.custom, message: We.message, promise: We.promise, dismiss: We.dismiss, loading: We.loading}, {getHistory: oS});
function iS(e, {insertAt: t} = {}) {
	if (typeof document > 'u') return;
	let n = document.head || document.getElementsByTagName('head')[0],
		r = document.createElement('style');
	(r.type = 'text/css'), t === 'top' && n.firstChild ? n.insertBefore(r, n.firstChild) : n.appendChild(r), r.styleSheet ? (r.styleSheet.cssText = e) : r.appendChild(document.createTextNode(e));
}
iS(`:where(html[dir="ltr"]),:where([data-sonner-toaster][dir="ltr"]){--toast-icon-margin-start: -3px;--toast-icon-margin-end: 4px;--toast-svg-margin-start: -1px;--toast-svg-margin-end: 0px;--toast-button-margin-start: auto;--toast-button-margin-end: 0;--toast-close-button-start: 0;--toast-close-button-end: unset;--toast-close-button-transform: translate(-35%, -35%)}:where(html[dir="rtl"]),:where([data-sonner-toaster][dir="rtl"]){--toast-icon-margin-start: 4px;--toast-icon-margin-end: -3px;--toast-svg-margin-start: 0px;--toast-svg-margin-end: -1px;--toast-button-margin-start: 0;--toast-button-margin-end: auto;--toast-close-button-start: unset;--toast-close-button-end: 0;--toast-close-button-transform: translate(35%, -35%)}:where([data-sonner-toaster]){position:fixed;width:var(--width);font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica Neue,Arial,Noto Sans,sans-serif,Apple Color Emoji,Segoe UI Emoji,Segoe UI Symbol,Noto Color Emoji;--gray1: hsl(0, 0%, 99%);--gray2: hsl(0, 0%, 97.3%);--gray3: hsl(0, 0%, 95.1%);--gray4: hsl(0, 0%, 93%);--gray5: hsl(0, 0%, 90.9%);--gray6: hsl(0, 0%, 88.7%);--gray7: hsl(0, 0%, 85.8%);--gray8: hsl(0, 0%, 78%);--gray9: hsl(0, 0%, 56.1%);--gray10: hsl(0, 0%, 52.3%);--gray11: hsl(0, 0%, 43.5%);--gray12: hsl(0, 0%, 9%);--border-radius: 8px;box-sizing:border-box;padding:0;margin:0;list-style:none;outline:none;z-index:999999999}:where([data-sonner-toaster][data-x-position="right"]){right:max(var(--offset),env(safe-area-inset-right))}:where([data-sonner-toaster][data-x-position="left"]){left:max(var(--offset),env(safe-area-inset-left))}:where([data-sonner-toaster][data-x-position="center"]){left:50%;transform:translate(-50%)}:where([data-sonner-toaster][data-y-position="top"]){top:max(var(--offset),env(safe-area-inset-top))}:where([data-sonner-toaster][data-y-position="bottom"]){bottom:max(var(--offset),env(safe-area-inset-bottom))}:where([data-sonner-toast]){--y: translateY(100%);--lift-amount: calc(var(--lift) * var(--gap));z-index:var(--z-index);position:absolute;opacity:0;transform:var(--y);filter:blur(0);touch-action:none;transition:transform .4s,opacity .4s,height .4s,box-shadow .2s;box-sizing:border-box;outline:none;overflow-wrap:anywhere}:where([data-sonner-toast][data-styled="true"]){padding:16px;background:var(--normal-bg);border:1px solid var(--normal-border);color:var(--normal-text);border-radius:var(--border-radius);box-shadow:0 4px 12px #0000001a;width:var(--width);font-size:13px;display:flex;align-items:center;gap:6px}:where([data-sonner-toast]:focus-visible){box-shadow:0 4px 12px #0000001a,0 0 0 2px #0003}:where([data-sonner-toast][data-y-position="top"]){top:0;--y: translateY(-100%);--lift: 1;--lift-amount: calc(1 * var(--gap))}:where([data-sonner-toast][data-y-position="bottom"]){bottom:0;--y: translateY(100%);--lift: -1;--lift-amount: calc(var(--lift) * var(--gap))}:where([data-sonner-toast]) :where([data-description]){font-weight:400;line-height:1.4;color:inherit}:where([data-sonner-toast]) :where([data-title]){font-weight:500;line-height:1.5;color:inherit}:where([data-sonner-toast]) :where([data-icon]){display:flex;height:16px;width:16px;position:relative;justify-content:flex-start;align-items:center;flex-shrink:0;margin-left:var(--toast-icon-margin-start);margin-right:var(--toast-icon-margin-end)}:where([data-sonner-toast][data-promise="true"]) :where([data-icon])>svg{opacity:0;transform:scale(.8);transform-origin:center;animation:sonner-fade-in .3s ease forwards}:where([data-sonner-toast]) :where([data-icon])>*{flex-shrink:0}:where([data-sonner-toast]) :where([data-icon]) svg{margin-left:var(--toast-svg-margin-start);margin-right:var(--toast-svg-margin-end)}:where([data-sonner-toast]) :where([data-content]){display:flex;flex-direction:column;gap:2px}[data-sonner-toast][data-styled=true] [data-button]{border-radius:4px;padding-left:8px;padding-right:8px;height:24px;font-size:12px;color:var(--normal-bg);background:var(--normal-text);margin-left:var(--toast-button-margin-start);margin-right:var(--toast-button-margin-end);border:none;cursor:pointer;outline:none;display:flex;align-items:center;flex-shrink:0;transition:opacity .4s,box-shadow .2s}:where([data-sonner-toast]) :where([data-button]):focus-visible{box-shadow:0 0 0 2px #0006}:where([data-sonner-toast]) :where([data-button]):first-of-type{margin-left:var(--toast-button-margin-start);margin-right:var(--toast-button-margin-end)}:where([data-sonner-toast]) :where([data-cancel]){color:var(--normal-text);background:rgba(0,0,0,.08)}:where([data-sonner-toast][data-theme="dark"]) :where([data-cancel]){background:rgba(255,255,255,.3)}:where([data-sonner-toast]) :where([data-close-button]){position:absolute;left:var(--toast-close-button-start);right:var(--toast-close-button-end);top:0;height:20px;width:20px;display:flex;justify-content:center;align-items:center;padding:0;background:var(--gray1);color:var(--gray12);border:1px solid var(--gray4);transform:var(--toast-close-button-transform);border-radius:50%;cursor:pointer;z-index:1;transition:opacity .1s,background .2s,border-color .2s}:where([data-sonner-toast]) :where([data-close-button]):focus-visible{box-shadow:0 4px 12px #0000001a,0 0 0 2px #0003}:where([data-sonner-toast]) :where([data-disabled="true"]){cursor:not-allowed}:where([data-sonner-toast]):hover :where([data-close-button]):hover{background:var(--gray2);border-color:var(--gray5)}:where([data-sonner-toast][data-swiping="true"]):before{content:"";position:absolute;left:0;right:0;height:100%;z-index:-1}:where([data-sonner-toast][data-y-position="top"][data-swiping="true"]):before{bottom:50%;transform:scaleY(3) translateY(50%)}:where([data-sonner-toast][data-y-position="bottom"][data-swiping="true"]):before{top:50%;transform:scaleY(3) translateY(-50%)}:where([data-sonner-toast][data-swiping="false"][data-removed="true"]):before{content:"";position:absolute;inset:0;transform:scaleY(2)}:where([data-sonner-toast]):after{content:"";position:absolute;left:0;height:calc(var(--gap) + 1px);bottom:100%;width:100%}:where([data-sonner-toast][data-mounted="true"]){--y: translateY(0);opacity:1}:where([data-sonner-toast][data-expanded="false"][data-front="false"]){--scale: var(--toasts-before) * .05 + 1;--y: translateY(calc(var(--lift-amount) * var(--toasts-before))) scale(calc(-1 * var(--scale)));height:var(--front-toast-height)}:where([data-sonner-toast])>*{transition:opacity .4s}:where([data-sonner-toast][data-expanded="false"][data-front="false"][data-styled="true"])>*{opacity:0}:where([data-sonner-toast][data-visible="false"]){opacity:0;pointer-events:none}:where([data-sonner-toast][data-mounted="true"][data-expanded="true"]){--y: translateY(calc(var(--lift) * var(--offset)));height:var(--initial-height)}:where([data-sonner-toast][data-removed="true"][data-front="true"][data-swipe-out="false"]){--y: translateY(calc(var(--lift) * -100%));opacity:0}:where([data-sonner-toast][data-removed="true"][data-front="false"][data-swipe-out="false"][data-expanded="true"]){--y: translateY(calc(var(--lift) * var(--offset) + var(--lift) * -100%));opacity:0}:where([data-sonner-toast][data-removed="true"][data-front="false"][data-swipe-out="false"][data-expanded="false"]){--y: translateY(40%);opacity:0;transition:transform .5s,opacity .2s}:where([data-sonner-toast][data-removed="true"][data-front="false"]):before{height:calc(var(--initial-height) + 20%)}[data-sonner-toast][data-swiping=true]{transform:var(--y) translateY(var(--swipe-amount, 0px));transition:none}[data-sonner-toast][data-swipe-out=true][data-y-position=bottom],[data-sonner-toast][data-swipe-out=true][data-y-position=top]{animation:swipe-out .2s ease-out forwards}@keyframes swipe-out{0%{transform:translateY(calc(var(--lift) * var(--offset) + var(--swipe-amount)));opacity:1}to{transform:translateY(calc(var(--lift) * var(--offset) + var(--swipe-amount) + var(--lift) * -100%));opacity:0}}@media (max-width: 600px){[data-sonner-toaster]{position:fixed;--mobile-offset: 16px;right:var(--mobile-offset);left:var(--mobile-offset);width:100%}[data-sonner-toaster] [data-sonner-toast]{left:0;right:0;width:calc(100% - var(--mobile-offset) * 2)}[data-sonner-toaster][data-x-position=left]{left:var(--mobile-offset)}[data-sonner-toaster][data-y-position=bottom]{bottom:20px}[data-sonner-toaster][data-y-position=top]{top:20px}[data-sonner-toaster][data-x-position=center]{left:var(--mobile-offset);right:var(--mobile-offset);transform:none}}[data-sonner-toaster][data-theme=light]{--normal-bg: #fff;--normal-border: var(--gray4);--normal-text: var(--gray12);--success-bg: hsl(143, 85%, 96%);--success-border: hsl(145, 92%, 91%);--success-text: hsl(140, 100%, 27%);--info-bg: hsl(208, 100%, 97%);--info-border: hsl(221, 91%, 91%);--info-text: hsl(210, 92%, 45%);--warning-bg: hsl(49, 100%, 97%);--warning-border: hsl(49, 91%, 91%);--warning-text: hsl(31, 92%, 45%);--error-bg: hsl(359, 100%, 97%);--error-border: hsl(359, 100%, 94%);--error-text: hsl(360, 100%, 45%)}[data-sonner-toaster][data-theme=light] [data-sonner-toast][data-invert=true]{--normal-bg: #000;--normal-border: hsl(0, 0%, 20%);--normal-text: var(--gray1)}[data-sonner-toaster][data-theme=dark] [data-sonner-toast][data-invert=true]{--normal-bg: #fff;--normal-border: var(--gray3);--normal-text: var(--gray12)}[data-sonner-toaster][data-theme=dark]{--normal-bg: #000;--normal-border: hsl(0, 0%, 20%);--normal-text: var(--gray1);--success-bg: hsl(150, 100%, 6%);--success-border: hsl(147, 100%, 12%);--success-text: hsl(150, 86%, 65%);--info-bg: hsl(215, 100%, 6%);--info-border: hsl(223, 100%, 12%);--info-text: hsl(216, 87%, 65%);--warning-bg: hsl(64, 100%, 6%);--warning-border: hsl(60, 100%, 12%);--warning-text: hsl(46, 87%, 65%);--error-bg: hsl(358, 76%, 10%);--error-border: hsl(357, 89%, 16%);--error-text: hsl(358, 100%, 81%)}[data-rich-colors=true][data-sonner-toast][data-type=success],[data-rich-colors=true][data-sonner-toast][data-type=success] [data-close-button]{background:var(--success-bg);border-color:var(--success-border);color:var(--success-text)}[data-rich-colors=true][data-sonner-toast][data-type=info],[data-rich-colors=true][data-sonner-toast][data-type=info] [data-close-button]{background:var(--info-bg);border-color:var(--info-border);color:var(--info-text)}[data-rich-colors=true][data-sonner-toast][data-type=warning],[data-rich-colors=true][data-sonner-toast][data-type=warning] [data-close-button]{background:var(--warning-bg);border-color:var(--warning-border);color:var(--warning-text)}[data-rich-colors=true][data-sonner-toast][data-type=error],[data-rich-colors=true][data-sonner-toast][data-type=error] [data-close-button]{background:var(--error-bg);border-color:var(--error-border);color:var(--error-text)}.sonner-loading-wrapper{--size: 16px;height:var(--size);width:var(--size);position:absolute;inset:0;z-index:10}.sonner-loading-wrapper[data-visible=false]{transform-origin:center;animation:sonner-fade-out .2s ease forwards}.sonner-spinner{position:relative;top:50%;left:50%;height:var(--size);width:var(--size)}.sonner-loading-bar{animation:sonner-spin 1.2s linear infinite;background:var(--gray11);border-radius:6px;height:8%;left:-10%;position:absolute;top:-3.9%;width:24%}.sonner-loading-bar:nth-child(1){animation-delay:-1.2s;transform:rotate(.0001deg) translate(146%)}.sonner-loading-bar:nth-child(2){animation-delay:-1.1s;transform:rotate(30deg) translate(146%)}.sonner-loading-bar:nth-child(3){animation-delay:-1s;transform:rotate(60deg) translate(146%)}.sonner-loading-bar:nth-child(4){animation-delay:-.9s;transform:rotate(90deg) translate(146%)}.sonner-loading-bar:nth-child(5){animation-delay:-.8s;transform:rotate(120deg) translate(146%)}.sonner-loading-bar:nth-child(6){animation-delay:-.7s;transform:rotate(150deg) translate(146%)}.sonner-loading-bar:nth-child(7){animation-delay:-.6s;transform:rotate(180deg) translate(146%)}.sonner-loading-bar:nth-child(8){animation-delay:-.5s;transform:rotate(210deg) translate(146%)}.sonner-loading-bar:nth-child(9){animation-delay:-.4s;transform:rotate(240deg) translate(146%)}.sonner-loading-bar:nth-child(10){animation-delay:-.3s;transform:rotate(270deg) translate(146%)}.sonner-loading-bar:nth-child(11){animation-delay:-.2s;transform:rotate(300deg) translate(146%)}.sonner-loading-bar:nth-child(12){animation-delay:-.1s;transform:rotate(330deg) translate(146%)}@keyframes sonner-fade-in{0%{opacity:0;transform:scale(.8)}to{opacity:1;transform:scale(1)}}@keyframes sonner-fade-out{0%{opacity:1;transform:scale(1)}to{opacity:0;transform:scale(.8)}}@keyframes sonner-spin{0%{opacity:1}to{opacity:.15}}@media (prefers-reduced-motion){[data-sonner-toast],[data-sonner-toast]>*,.sonner-loading-bar{transition:none!important;animation:none!important}}.sonner-loader{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);transform-origin:center;transition:opacity .2s,transform .2s}.sonner-loader[data-visible=false]{opacity:0;transform:scale(.8) translate(-50%,-50%)}
`);
function hi(e) {
	return e.label !== void 0;
}
var sS = 3,
	lS = '32px',
	aS = 4e3,
	uS = 356,
	cS = 14,
	dS = 20,
	fS = 200;
function pS(...e) {
	return e.filter(Boolean).join(' ');
}
var mS = (e) => {
	var t, n, r, o, i, s, l, a, u, f;
	let {
			invert: d,
			toast: c,
			unstyled: g,
			interacting: w,
			setHeights: y,
			visibleToasts: S,
			heights: m,
			index: h,
			toasts: x,
			expanded: E,
			removeToast: C,
			defaultRichColors: T,
			closeButton: k,
			style: N,
			cancelButtonStyle: D,
			actionButtonStyle: M,
			className: z = '',
			descriptionClassName: R = '',
			duration: $,
			position: U,
			gap: Q,
			loadingIcon: F,
			expandByDefault: X,
			classNames: b,
			icons: P,
			closeButtonAriaLabel: j = 'Close toast',
			pauseWhenPageIsHidden: O,
			cn: L,
		} = e,
		[Y, ie] = A.useState(!1),
		[Ne, K] = A.useState(!1),
		[xe, st] = A.useState(!1),
		[Se, lt] = A.useState(!1),
		[Go, Wn] = A.useState(0),
		[Sn, Fr] = A.useState(0),
		Ko = A.useRef(null),
		Vt = A.useRef(null),
		qs = h === 0,
		Js = h + 1 <= S,
		be = c.type,
		Vn = c.dismissible !== !1,
		Nv = c.className || '',
		_v = c.descriptionClassName || '',
		Yo = A.useMemo(() => m.findIndex((W) => W.toastId === c.id) || 0, [m, c.id]),
		Rv = A.useMemo(() => {
			var W;
			return (W = c.closeButton) != null ? W : k;
		}, [c.closeButton, k]),
		bc = A.useMemo(() => c.duration || $ || aS, [c.duration, $]),
		el = A.useRef(0),
		Gn = A.useRef(0),
		kc = A.useRef(0),
		Kn = A.useRef(null),
		[Tc, Av] = U.split('-'),
		Pc = A.useMemo(() => m.reduce((W, te, J) => (J >= Yo ? W : W + te.height), 0), [m, Yo]),
		Nc = J1(),
		Mv = c.invert || d,
		tl = be === 'loading';
	(Gn.current = A.useMemo(() => Yo * Q + Pc, [Yo, Pc])),
		A.useEffect(() => {
			ie(!0);
		}, []),
		A.useLayoutEffect(() => {
			if (!Y) return;
			let W = Vt.current,
				te = W.style.height;
			W.style.height = 'auto';
			let J = W.getBoundingClientRect().height;
			(W.style.height = te), Fr(J), y((xt) => (xt.find((St) => St.toastId === c.id) ? xt.map((St) => (St.toastId === c.id ? {...St, height: J} : St)) : [{toastId: c.id, height: J, position: c.position}, ...xt]));
		}, [Y, c.title, c.description, y, c.id]);
	let Gt = A.useCallback(() => {
		K(!0),
			Wn(Gn.current),
			y((W) => W.filter((te) => te.toastId !== c.id)),
			setTimeout(() => {
				C(c);
			}, fS);
	}, [c, C, y, Gn]);
	A.useEffect(() => {
		if ((c.promise && be === 'loading') || c.duration === 1 / 0 || c.type === 'loading') return;
		let W,
			te = bc;
		return (
			E || w || (O && Nc)
				? (() => {
						if (kc.current < el.current) {
							let J = new Date().getTime() - el.current;
							te = te - J;
						}
						kc.current = new Date().getTime();
				  })()
				: te !== 1 / 0 &&
				  ((el.current = new Date().getTime()),
				  (W = setTimeout(() => {
						var J;
						(J = c.onAutoClose) == null || J.call(c, c), Gt();
				  }, te))),
			() => clearTimeout(W)
		);
	}, [E, w, X, c, bc, Gt, c.promise, be, O, Nc]),
		A.useEffect(() => {
			let W = Vt.current;
			if (W) {
				let te = W.getBoundingClientRect().height;
				return Fr(te), y((J) => [{toastId: c.id, height: te, position: c.position}, ...J]), () => y((J) => J.filter((xt) => xt.toastId !== c.id));
			}
		}, [y, c.id]),
		A.useEffect(() => {
			c.delete && Gt();
		}, [Gt, c.delete]);
	function Ov() {
		return P != null && P.loading
			? A.createElement('div', {className: 'sonner-loader', 'data-visible': be === 'loading'}, P.loading)
			: F
			? A.createElement('div', {className: 'sonner-loader', 'data-visible': be === 'loading'}, F)
			: A.createElement(Y1, {visible: be === 'loading'});
	}
	return A.createElement(
		'li',
		{
			'aria-live': c.important ? 'assertive' : 'polite',
			'aria-atomic': 'true',
			role: 'status',
			tabIndex: 0,
			ref: Vt,
			className: L(z, Nv, b == null ? void 0 : b.toast, (t = c == null ? void 0 : c.classNames) == null ? void 0 : t.toast, b == null ? void 0 : b.default, b == null ? void 0 : b[be], (n = c == null ? void 0 : c.classNames) == null ? void 0 : n[be]),
			'data-sonner-toast': '',
			'data-rich-colors': (r = c.richColors) != null ? r : T,
			'data-styled': !(c.jsx || c.unstyled || g),
			'data-mounted': Y,
			'data-promise': !!c.promise,
			'data-removed': Ne,
			'data-visible': Js,
			'data-y-position': Tc,
			'data-x-position': Av,
			'data-index': h,
			'data-front': qs,
			'data-swiping': xe,
			'data-dismissible': Vn,
			'data-type': be,
			'data-invert': Mv,
			'data-swipe-out': Se,
			'data-expanded': !!(E || (X && Y)),
			style: {'--index': h, '--toasts-before': h, '--z-index': x.length - h, '--offset': `${Ne ? Go : Gn.current}px`, '--initial-height': X ? 'auto' : `${Sn}px`, ...N, ...c.style},
			onPointerDown: (W) => {
				tl || !Vn || ((Ko.current = new Date()), Wn(Gn.current), W.target.setPointerCapture(W.pointerId), W.target.tagName !== 'BUTTON' && (st(!0), (Kn.current = {x: W.clientX, y: W.clientY})));
			},
			onPointerUp: () => {
				var W, te, J, xt;
				if (Se || !Vn) return;
				Kn.current = null;
				let St = Number(((W = Vt.current) == null ? void 0 : W.style.getPropertyValue('--swipe-amount').replace('px', '')) || 0),
					Qo = new Date().getTime() - ((te = Ko.current) == null ? void 0 : te.getTime()),
					jv = Math.abs(St) / Qo;
				if (Math.abs(St) >= dS || jv > 0.11) {
					Wn(Gn.current), (J = c.onDismiss) == null || J.call(c, c), Gt(), lt(!0);
					return;
				}
				(xt = Vt.current) == null || xt.style.setProperty('--swipe-amount', '0px'), st(!1);
			},
			onPointerMove: (W) => {
				var te;
				if (!Kn.current || !Vn) return;
				let J = W.clientY - Kn.current.y,
					xt = W.clientX - Kn.current.x,
					St = (Tc === 'top' ? Math.min : Math.max)(0, J),
					Qo = W.pointerType === 'touch' ? 10 : 2;
				Math.abs(St) > Qo ? (te = Vt.current) == null || te.style.setProperty('--swipe-amount', `${J}px`) : Math.abs(xt) > Qo && (Kn.current = null);
			},
		},
		Rv && !c.jsx
			? A.createElement(
					'button',
					{
						'aria-label': j,
						'data-disabled': tl,
						'data-close-button': !0,
						onClick:
							tl || !Vn
								? () => {}
								: () => {
										var W;
										Gt(), (W = c.onDismiss) == null || W.call(c, c);
								  },
						className: L(b == null ? void 0 : b.closeButton, (o = c == null ? void 0 : c.classNames) == null ? void 0 : o.closeButton),
					},
					A.createElement(
						'svg',
						{xmlns: 'http://www.w3.org/2000/svg', width: '12', height: '12', viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: '1.5', strokeLinecap: 'round', strokeLinejoin: 'round'},
						A.createElement('line', {x1: '18', y1: '6', x2: '6', y2: '18'}),
						A.createElement('line', {x1: '6', y1: '6', x2: '18', y2: '18'})
					)
			  )
			: null,
		c.jsx || A.isValidElement(c.title)
			? c.jsx || c.title
			: A.createElement(
					A.Fragment,
					null,
					be || c.icon || c.promise
						? A.createElement(
								'div',
								{'data-icon': '', className: L(b == null ? void 0 : b.icon, (i = c == null ? void 0 : c.classNames) == null ? void 0 : i.icon)},
								c.promise || (c.type === 'loading' && !c.icon) ? c.icon || Ov() : null,
								c.type !== 'loading' ? c.icon || (P == null ? void 0 : P[be]) || G1(be) : null
						  )
						: null,
					A.createElement(
						'div',
						{'data-content': '', className: L(b == null ? void 0 : b.content, (s = c == null ? void 0 : c.classNames) == null ? void 0 : s.content)},
						A.createElement('div', {'data-title': '', className: L(b == null ? void 0 : b.title, (l = c == null ? void 0 : c.classNames) == null ? void 0 : l.title)}, c.title),
						c.description ? A.createElement('div', {'data-description': '', className: L(R, _v, b == null ? void 0 : b.description, (a = c == null ? void 0 : c.classNames) == null ? void 0 : a.description)}, c.description) : null
					),
					A.isValidElement(c.cancel)
						? c.cancel
						: c.cancel && hi(c.cancel)
						? A.createElement(
								'button',
								{
									'data-button': !0,
									'data-cancel': !0,
									style: c.cancelButtonStyle || D,
									onClick: (W) => {
										var te, J;
										hi(c.cancel) && Vn && ((J = (te = c.cancel).onClick) == null || J.call(te, W), Gt());
									},
									className: L(b == null ? void 0 : b.cancelButton, (u = c == null ? void 0 : c.classNames) == null ? void 0 : u.cancelButton),
								},
								c.cancel.label
						  )
						: null,
					A.isValidElement(c.action)
						? c.action
						: c.action && hi(c.action)
						? A.createElement(
								'button',
								{
									'data-button': !0,
									'data-action': !0,
									style: c.actionButtonStyle || M,
									onClick: (W) => {
										var te, J;
										hi(c.action) && (W.defaultPrevented || ((J = (te = c.action).onClick) == null || J.call(te, W), Gt()));
									},
									className: L(b == null ? void 0 : b.actionButton, (f = c == null ? void 0 : c.classNames) == null ? void 0 : f.actionButton),
								},
								c.action.label
						  )
						: null
			  )
	);
};
function ef() {
	if (typeof window > 'u' || typeof document > 'u') return 'ltr';
	let e = document.documentElement.getAttribute('dir');
	return e === 'auto' || !e ? window.getComputedStyle(document.documentElement).direction : e;
}
var hS = (e) => {
	let {
			invert: t,
			position: n = 'bottom-right',
			hotkey: r = ['altKey', 'KeyT'],
			expand: o,
			closeButton: i,
			className: s,
			offset: l,
			theme: a = 'light',
			richColors: u,
			duration: f,
			style: d,
			visibleToasts: c = sS,
			toastOptions: g,
			dir: w = ef(),
			gap: y = cS,
			loadingIcon: S,
			icons: m,
			containerAriaLabel: h = 'Notifications',
			pauseWhenPageIsHidden: x,
			cn: E = pS,
		} = e,
		[C, T] = A.useState([]),
		k = A.useMemo(() => Array.from(new Set([n].concat(C.filter((O) => O.position).map((O) => O.position)))), [C, n]),
		[N, D] = A.useState([]),
		[M, z] = A.useState(!1),
		[R, $] = A.useState(!1),
		[U, Q] = A.useState(a !== 'system' ? a : typeof window < 'u' && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'),
		F = A.useRef(null),
		X = r.join('+').replace(/Key/g, '').replace(/Digit/g, ''),
		b = A.useRef(null),
		P = A.useRef(!1),
		j = A.useCallback(
			(O) => {
				var L;
				((L = C.find((Y) => Y.id === O.id)) != null && L.delete) || We.dismiss(O.id), T((Y) => Y.filter(({id: ie}) => ie !== O.id));
			},
			[C]
		);
	return (
		A.useEffect(
			() =>
				We.subscribe((O) => {
					if (O.dismiss) {
						T((L) => L.map((Y) => (Y.id === O.id ? {...Y, delete: !0} : Y)));
						return;
					}
					setTimeout(() => {
						jm.flushSync(() => {
							T((L) => {
								let Y = L.findIndex((ie) => ie.id === O.id);
								return Y !== -1 ? [...L.slice(0, Y), {...L[Y], ...O}, ...L.slice(Y + 1)] : [O, ...L];
							});
						});
					});
				}),
			[]
		),
		A.useEffect(() => {
			if (a !== 'system') {
				Q(a);
				return;
			}
			a === 'system' && (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? Q('dark') : Q('light')),
				typeof window < 'u' &&
					window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', ({matches: O}) => {
						Q(O ? 'dark' : 'light');
					});
		}, [a]),
		A.useEffect(() => {
			C.length <= 1 && z(!1);
		}, [C]),
		A.useEffect(() => {
			let O = (L) => {
				var Y, ie;
				r.every((Ne) => L[Ne] || L.code === Ne) && (z(!0), (Y = F.current) == null || Y.focus()), L.code === 'Escape' && (document.activeElement === F.current || ((ie = F.current) != null && ie.contains(document.activeElement))) && z(!1);
			};
			return document.addEventListener('keydown', O), () => document.removeEventListener('keydown', O);
		}, [r]),
		A.useEffect(() => {
			if (F.current)
				return () => {
					b.current && (b.current.focus({preventScroll: !0}), (b.current = null), (P.current = !1));
				};
		}, [F.current]),
		C.length
			? A.createElement(
					'section',
					{'aria-label': `${h} ${X}`, tabIndex: -1},
					k.map((O, L) => {
						var Y;
						let [ie, Ne] = O.split('-');
						return A.createElement(
							'ol',
							{
								key: O,
								dir: w === 'auto' ? ef() : w,
								tabIndex: -1,
								ref: F,
								className: s,
								'data-sonner-toaster': !0,
								'data-theme': U,
								'data-y-position': ie,
								'data-x-position': Ne,
								style: {'--front-toast-height': `${((Y = N[0]) == null ? void 0 : Y.height) || 0}px`, '--offset': typeof l == 'number' ? `${l}px` : l || lS, '--width': `${uS}px`, '--gap': `${y}px`, ...d},
								onBlur: (K) => {
									P.current && !K.currentTarget.contains(K.relatedTarget) && ((P.current = !1), b.current && (b.current.focus({preventScroll: !0}), (b.current = null)));
								},
								onFocus: (K) => {
									(K.target instanceof HTMLElement && K.target.dataset.dismissible === 'false') || P.current || ((P.current = !0), (b.current = K.relatedTarget));
								},
								onMouseEnter: () => z(!0),
								onMouseMove: () => z(!0),
								onMouseLeave: () => {
									R || z(!1);
								},
								onPointerDown: (K) => {
									(K.target instanceof HTMLElement && K.target.dataset.dismissible === 'false') || $(!0);
								},
								onPointerUp: () => $(!1),
							},
							C.filter((K) => (!K.position && L === 0) || K.position === O).map((K, xe) => {
								var st, Se;
								return A.createElement(mS, {
									key: K.id,
									icons: m,
									index: xe,
									toast: K,
									defaultRichColors: u,
									duration: (st = g == null ? void 0 : g.duration) != null ? st : f,
									className: g == null ? void 0 : g.className,
									descriptionClassName: g == null ? void 0 : g.descriptionClassName,
									invert: t,
									visibleToasts: c,
									closeButton: (Se = g == null ? void 0 : g.closeButton) != null ? Se : i,
									interacting: R,
									position: O,
									style: g == null ? void 0 : g.style,
									unstyled: g == null ? void 0 : g.unstyled,
									classNames: g == null ? void 0 : g.classNames,
									cancelButtonStyle: g == null ? void 0 : g.cancelButtonStyle,
									actionButtonStyle: g == null ? void 0 : g.actionButtonStyle,
									removeToast: j,
									toasts: C.filter((lt) => lt.position == K.position),
									heights: N.filter((lt) => lt.position == K.position),
									setHeights: D,
									expandByDefault: o,
									gap: y,
									loadingIcon: S,
									expanded: M,
									pauseWhenPageIsHidden: x,
									cn: E,
								});
							})
						);
					})
			  )
			: null
	);
};
function gS(e, t = []) {
	let n = [];
	function r(i, s) {
		const l = p.createContext(s),
			a = n.length;
		n = [...n, s];
		function u(d) {
			const {scope: c, children: g, ...w} = d,
				y = (c == null ? void 0 : c[e][a]) || l,
				S = p.useMemo(() => w, Object.values(w));
			return v.jsx(y.Provider, {value: S, children: g});
		}
		function f(d, c) {
			const g = (c == null ? void 0 : c[e][a]) || l,
				w = p.useContext(g);
			if (w) return w;
			if (s !== void 0) return s;
			throw new Error(`\`${d}\` must be used within \`${i}\``);
		}
		return (u.displayName = i + 'Provider'), [u, f];
	}
	const o = () => {
		const i = n.map((s) => p.createContext(s));
		return function (l) {
			const a = (l == null ? void 0 : l[e]) || i;
			return p.useMemo(() => ({[`__scope${e}`]: {...l, [e]: a}}), [l, a]);
		};
	};
	return (o.scopeName = e), [r, vS(o, ...t)];
}
function vS(...e) {
	const t = e[0];
	if (e.length === 1) return t;
	const n = () => {
		const r = e.map((o) => ({useScope: o(), scopeName: o.scopeName}));
		return function (i) {
			const s = r.reduce((l, {useScope: a, scopeName: u}) => {
				const d = a(i)[`__scope${u}`];
				return {...l, ...d};
			}, {});
			return p.useMemo(() => ({[`__scope${t.scopeName}`]: s}), [s]);
		};
	};
	return (n.scopeName = t.scopeName), n;
}
function Nh(e) {
	const t = e + 'CollectionProvider',
		[n, r] = gS(t),
		[o, i] = n(t, {collectionRef: {current: null}, itemMap: new Map()}),
		s = (g) => {
			const {scope: w, children: y} = g,
				S = A.useRef(null),
				m = A.useRef(new Map()).current;
			return v.jsx(o, {scope: w, itemMap: m, collectionRef: S, children: y});
		};
	s.displayName = t;
	const l = e + 'CollectionSlot',
		a = A.forwardRef((g, w) => {
			const {scope: y, children: S} = g,
				m = i(l, y),
				h = we(w, m.collectionRef);
			return v.jsx(Dn, {ref: h, children: S});
		});
	a.displayName = l;
	const u = e + 'CollectionItemSlot',
		f = 'data-radix-collection-item',
		d = A.forwardRef((g, w) => {
			const {scope: y, children: S, ...m} = g,
				h = A.useRef(null),
				x = we(w, h),
				E = i(u, y);
			return A.useEffect(() => (E.itemMap.set(h, {ref: h, ...m}), () => void E.itemMap.delete(h))), v.jsx(Dn, {[f]: '', ref: x, children: S});
		});
	d.displayName = u;
	function c(g) {
		const w = i(e + 'CollectionConsumer', g);
		return A.useCallback(() => {
			const S = w.collectionRef.current;
			if (!S) return [];
			const m = Array.from(S.querySelectorAll(`[${f}]`));
			return Array.from(w.itemMap.values()).sort((E, C) => m.indexOf(E.ref.current) - m.indexOf(C.ref.current));
		}, [w.collectionRef, w.itemMap]);
	}
	return [{Provider: s, Slot: a, ItemSlot: d}, c, r];
}
var yS = p.createContext(void 0);
function _h(e) {
	const t = p.useContext(yS);
	return e || t || 'ltr';
}
var Ml = 0;
function Rh() {
	p.useEffect(() => {
		const e = document.querySelectorAll('[data-radix-focus-guard]');
		return (
			document.body.insertAdjacentElement('afterbegin', e[0] ?? tf()),
			document.body.insertAdjacentElement('beforeend', e[1] ?? tf()),
			Ml++,
			() => {
				Ml === 1 && document.querySelectorAll('[data-radix-focus-guard]').forEach((t) => t.remove()), Ml--;
			}
		);
	}, []);
}
function tf() {
	const e = document.createElement('span');
	return e.setAttribute('data-radix-focus-guard', ''), (e.tabIndex = 0), (e.style.outline = 'none'), (e.style.opacity = '0'), (e.style.position = 'fixed'), (e.style.pointerEvents = 'none'), e;
}
var Ol = 'focusScope.autoFocusOnMount',
	jl = 'focusScope.autoFocusOnUnmount',
	nf = {bubbles: !1, cancelable: !0},
	wS = 'FocusScope',
	lc = p.forwardRef((e, t) => {
		const {loop: n = !1, trapped: r = !1, onMountAutoFocus: o, onUnmountAutoFocus: i, ...s} = e,
			[l, a] = p.useState(null),
			u = gt(o),
			f = gt(i),
			d = p.useRef(null),
			c = we(t, (y) => a(y)),
			g = p.useRef({
				paused: !1,
				pause() {
					this.paused = !0;
				},
				resume() {
					this.paused = !1;
				},
			}).current;
		p.useEffect(() => {
			if (r) {
				let y = function (x) {
						if (g.paused || !l) return;
						const E = x.target;
						l.contains(E) ? (d.current = E) : Xt(d.current, {select: !0});
					},
					S = function (x) {
						if (g.paused || !l) return;
						const E = x.relatedTarget;
						E !== null && (l.contains(E) || Xt(d.current, {select: !0}));
					},
					m = function (x) {
						if (document.activeElement === document.body) for (const C of x) C.removedNodes.length > 0 && Xt(l);
					};
				document.addEventListener('focusin', y), document.addEventListener('focusout', S);
				const h = new MutationObserver(m);
				return (
					l && h.observe(l, {childList: !0, subtree: !0}),
					() => {
						document.removeEventListener('focusin', y), document.removeEventListener('focusout', S), h.disconnect();
					}
				);
			}
		}, [r, l, g.paused]),
			p.useEffect(() => {
				if (l) {
					of.add(g);
					const y = document.activeElement;
					if (!l.contains(y)) {
						const m = new CustomEvent(Ol, nf);
						l.addEventListener(Ol, u), l.dispatchEvent(m), m.defaultPrevented || (xS(kS(Ah(l)), {select: !0}), document.activeElement === y && Xt(l));
					}
					return () => {
						l.removeEventListener(Ol, u),
							setTimeout(() => {
								const m = new CustomEvent(jl, nf);
								l.addEventListener(jl, f), l.dispatchEvent(m), m.defaultPrevented || Xt(y ?? document.body, {select: !0}), l.removeEventListener(jl, f), of.remove(g);
							}, 0);
					};
				}
			}, [l, u, f, g]);
		const w = p.useCallback(
			(y) => {
				if ((!n && !r) || g.paused) return;
				const S = y.key === 'Tab' && !y.altKey && !y.ctrlKey && !y.metaKey,
					m = document.activeElement;
				if (S && m) {
					const h = y.currentTarget,
						[x, E] = SS(h);
					x && E ? (!y.shiftKey && m === E ? (y.preventDefault(), n && Xt(x, {select: !0})) : y.shiftKey && m === x && (y.preventDefault(), n && Xt(E, {select: !0}))) : m === h && y.preventDefault();
				}
			},
			[n, r, g.paused]
		);
		return v.jsx(pe.div, {tabIndex: -1, ...s, ref: c, onKeyDown: w});
	});
lc.displayName = wS;
function xS(e, {select: t = !1} = {}) {
	const n = document.activeElement;
	for (const r of e) if ((Xt(r, {select: t}), document.activeElement !== n)) return;
}
function SS(e) {
	const t = Ah(e),
		n = rf(t, e),
		r = rf(t.reverse(), e);
	return [n, r];
}
function Ah(e) {
	const t = [],
		n = document.createTreeWalker(e, NodeFilter.SHOW_ELEMENT, {
			acceptNode: (r) => {
				const o = r.tagName === 'INPUT' && r.type === 'hidden';
				return r.disabled || r.hidden || o ? NodeFilter.FILTER_SKIP : r.tabIndex >= 0 ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
			},
		});
	for (; n.nextNode(); ) t.push(n.currentNode);
	return t;
}
function rf(e, t) {
	for (const n of e) if (!ES(n, {upTo: t})) return n;
}
function ES(e, {upTo: t}) {
	if (getComputedStyle(e).visibility === 'hidden') return !0;
	for (; e; ) {
		if (t !== void 0 && e === t) return !1;
		if (getComputedStyle(e).display === 'none') return !0;
		e = e.parentElement;
	}
	return !1;
}
function CS(e) {
	return e instanceof HTMLInputElement && 'select' in e;
}
function Xt(e, {select: t = !1} = {}) {
	if (e && e.focus) {
		const n = document.activeElement;
		e.focus({preventScroll: !0}), e !== n && CS(e) && t && e.select();
	}
}
var of = bS();
function bS() {
	let e = [];
	return {
		add(t) {
			const n = e[0];
			t !== n && (n == null || n.pause()), (e = sf(e, t)), e.unshift(t);
		},
		remove(t) {
			var n;
			(e = sf(e, t)), (n = e[0]) == null || n.resume();
		},
	};
}
function sf(e, t) {
	const n = [...e],
		r = n.indexOf(t);
	return r !== -1 && n.splice(r, 1), n;
}
function kS(e) {
	return e.filter((t) => t.tagName !== 'A');
}
function TS(e, t = []) {
	let n = [];
	function r(i, s) {
		const l = p.createContext(s),
			a = n.length;
		n = [...n, s];
		function u(d) {
			const {scope: c, children: g, ...w} = d,
				y = (c == null ? void 0 : c[e][a]) || l,
				S = p.useMemo(() => w, Object.values(w));
			return v.jsx(y.Provider, {value: S, children: g});
		}
		function f(d, c) {
			const g = (c == null ? void 0 : c[e][a]) || l,
				w = p.useContext(g);
			if (w) return w;
			if (s !== void 0) return s;
			throw new Error(`\`${d}\` must be used within \`${i}\``);
		}
		return (u.displayName = i + 'Provider'), [u, f];
	}
	const o = () => {
		const i = n.map((s) => p.createContext(s));
		return function (l) {
			const a = (l == null ? void 0 : l[e]) || i;
			return p.useMemo(() => ({[`__scope${e}`]: {...l, [e]: a}}), [l, a]);
		};
	};
	return (o.scopeName = e), [r, PS(o, ...t)];
}
function PS(...e) {
	const t = e[0];
	if (e.length === 1) return t;
	const n = () => {
		const r = e.map((o) => ({useScope: o(), scopeName: o.scopeName}));
		return function (i) {
			const s = r.reduce((l, {useScope: a, scopeName: u}) => {
				const d = a(i)[`__scope${u}`];
				return {...l, ...d};
			}, {});
			return p.useMemo(() => ({[`__scope${t.scopeName}`]: s}), [s]);
		};
	};
	return (n.scopeName = t.scopeName), n;
}
var Il = 'rovingFocusGroup.onEntryFocus',
	NS = {bubbles: !1, cancelable: !0},
	Ws = 'RovingFocusGroup',
	[Va, Mh, _S] = Nh(Ws),
	[RS, Oh] = TS(Ws, [_S]),
	[AS, MS] = RS(Ws),
	jh = p.forwardRef((e, t) => v.jsx(Va.Provider, {scope: e.__scopeRovingFocusGroup, children: v.jsx(Va.Slot, {scope: e.__scopeRovingFocusGroup, children: v.jsx(OS, {...e, ref: t})})}));
jh.displayName = Ws;
var OS = p.forwardRef((e, t) => {
		const {__scopeRovingFocusGroup: n, orientation: r, loop: o = !1, dir: i, currentTabStopId: s, defaultCurrentTabStopId: l, onCurrentTabStopIdChange: a, onEntryFocus: u, preventScrollOnEntryFocus: f = !1, ...d} = e,
			c = p.useRef(null),
			g = we(t, c),
			w = _h(i),
			[y = null, S] = zs({prop: s, defaultProp: l, onChange: a}),
			[m, h] = p.useState(!1),
			x = gt(u),
			E = Mh(n),
			C = p.useRef(!1),
			[T, k] = p.useState(0);
		return (
			p.useEffect(() => {
				const N = c.current;
				if (N) return N.addEventListener(Il, x), () => N.removeEventListener(Il, x);
			}, [x]),
			v.jsx(AS, {
				scope: n,
				orientation: r,
				dir: w,
				loop: o,
				currentTabStopId: y,
				onItemFocus: p.useCallback((N) => S(N), [S]),
				onItemShiftTab: p.useCallback(() => h(!0), []),
				onFocusableItemAdd: p.useCallback(() => k((N) => N + 1), []),
				onFocusableItemRemove: p.useCallback(() => k((N) => N - 1), []),
				children: v.jsx(pe.div, {
					tabIndex: m || T === 0 ? -1 : 0,
					'data-orientation': r,
					...d,
					ref: g,
					style: {outline: 'none', ...e.style},
					onMouseDown: B(e.onMouseDown, () => {
						C.current = !0;
					}),
					onFocus: B(e.onFocus, (N) => {
						const D = !C.current;
						if (N.target === N.currentTarget && D && !m) {
							const M = new CustomEvent(Il, NS);
							if ((N.currentTarget.dispatchEvent(M), !M.defaultPrevented)) {
								const z = E().filter((F) => F.focusable),
									R = z.find((F) => F.active),
									$ = z.find((F) => F.id === y),
									Q = [R, $, ...z].filter(Boolean).map((F) => F.ref.current);
								Lh(Q, f);
							}
						}
						C.current = !1;
					}),
					onBlur: B(e.onBlur, () => h(!1)),
				}),
			})
		);
	}),
	Ih = 'RovingFocusGroupItem',
	Dh = p.forwardRef((e, t) => {
		const {__scopeRovingFocusGroup: n, focusable: r = !0, active: o = !1, tabStopId: i, ...s} = e,
			l = _n(),
			a = i || l,
			u = MS(Ih, n),
			f = u.currentTabStopId === a,
			d = Mh(n),
			{onFocusableItemAdd: c, onFocusableItemRemove: g} = u;
		return (
			p.useEffect(() => {
				if (r) return c(), () => g();
			}, [r, c, g]),
			v.jsx(Va.ItemSlot, {
				scope: n,
				id: a,
				focusable: r,
				active: o,
				children: v.jsx(pe.span, {
					tabIndex: f ? 0 : -1,
					'data-orientation': u.orientation,
					...s,
					ref: t,
					onMouseDown: B(e.onMouseDown, (w) => {
						r ? u.onItemFocus(a) : w.preventDefault();
					}),
					onFocus: B(e.onFocus, () => u.onItemFocus(a)),
					onKeyDown: B(e.onKeyDown, (w) => {
						if (w.key === 'Tab' && w.shiftKey) {
							u.onItemShiftTab();
							return;
						}
						if (w.target !== w.currentTarget) return;
						const y = DS(w, u.orientation, u.dir);
						if (y !== void 0) {
							if (w.metaKey || w.ctrlKey || w.altKey || w.shiftKey) return;
							w.preventDefault();
							let m = d()
								.filter((h) => h.focusable)
								.map((h) => h.ref.current);
							if (y === 'last') m.reverse();
							else if (y === 'prev' || y === 'next') {
								y === 'prev' && m.reverse();
								const h = m.indexOf(w.currentTarget);
								m = u.loop ? LS(m, h + 1) : m.slice(h + 1);
							}
							setTimeout(() => Lh(m));
						}
					}),
				}),
			})
		);
	});
Dh.displayName = Ih;
var jS = {ArrowLeft: 'prev', ArrowUp: 'prev', ArrowRight: 'next', ArrowDown: 'next', PageUp: 'first', Home: 'first', PageDown: 'last', End: 'last'};
function IS(e, t) {
	return t !== 'rtl' ? e : e === 'ArrowLeft' ? 'ArrowRight' : e === 'ArrowRight' ? 'ArrowLeft' : e;
}
function DS(e, t, n) {
	const r = IS(e.key, n);
	if (!(t === 'vertical' && ['ArrowLeft', 'ArrowRight'].includes(r)) && !(t === 'horizontal' && ['ArrowUp', 'ArrowDown'].includes(r))) return jS[r];
}
function Lh(e, t = !1) {
	const n = document.activeElement;
	for (const r of e) if (r === n || (r.focus({preventScroll: t}), document.activeElement !== n)) return;
}
function LS(e, t) {
	return e.map((n, r) => e[(t + r) % e.length]);
}
var FS = jh,
	zS = Dh,
	$S = function (e) {
		if (typeof document > 'u') return null;
		var t = Array.isArray(e) ? e[0] : e;
		return t.ownerDocument.body;
	},
	Qn = new WeakMap(),
	gi = new WeakMap(),
	vi = {},
	Dl = 0,
	Fh = function (e) {
		return e && (e.host || Fh(e.parentNode));
	},
	BS = function (e, t) {
		return t
			.map(function (n) {
				if (e.contains(n)) return n;
				var r = Fh(n);
				return r && e.contains(r) ? r : (console.error('aria-hidden', n, 'in not contained inside', e, '. Doing nothing'), null);
			})
			.filter(function (n) {
				return !!n;
			});
	},
	US = function (e, t, n, r) {
		var o = BS(t, Array.isArray(e) ? e : [e]);
		vi[n] || (vi[n] = new WeakMap());
		var i = vi[n],
			s = [],
			l = new Set(),
			a = new Set(o),
			u = function (d) {
				!d || l.has(d) || (l.add(d), u(d.parentNode));
			};
		o.forEach(u);
		var f = function (d) {
			!d ||
				a.has(d) ||
				Array.prototype.forEach.call(d.children, function (c) {
					if (l.has(c)) f(c);
					else
						try {
							var g = c.getAttribute(r),
								w = g !== null && g !== 'false',
								y = (Qn.get(c) || 0) + 1,
								S = (i.get(c) || 0) + 1;
							Qn.set(c, y), i.set(c, S), s.push(c), y === 1 && w && gi.set(c, !0), S === 1 && c.setAttribute(n, 'true'), w || c.setAttribute(r, 'true');
						} catch (m) {
							console.error('aria-hidden: cannot operate on ', c, m);
						}
				});
		};
		return (
			f(t),
			l.clear(),
			Dl++,
			function () {
				s.forEach(function (d) {
					var c = Qn.get(d) - 1,
						g = i.get(d) - 1;
					Qn.set(d, c), i.set(d, g), c || (gi.has(d) || d.removeAttribute(r), gi.delete(d)), g || d.removeAttribute(n);
				}),
					Dl--,
					Dl || ((Qn = new WeakMap()), (Qn = new WeakMap()), (gi = new WeakMap()), (vi = {}));
			}
		);
	},
	zh = function (e, t, n) {
		n === void 0 && (n = 'data-aria-hidden');
		var r = Array.from(Array.isArray(e) ? e : [e]),
			o = $S(e);
		return o
			? (r.push.apply(r, Array.from(o.querySelectorAll('[aria-live]'))), US(r, o, n, 'aria-hiddenfgfg'))
			: function () {
					return null;
			  };
	},
	kt = function () {
		return (
			(kt =
				Object.assign ||
				function (t) {
					for (var n, r = 1, o = arguments.length; r < o; r++) {
						n = arguments[r];
						for (var i in n) Object.prototype.hasOwnProperty.call(n, i) && (t[i] = n[i]);
					}
					return t;
				}),
			kt.apply(this, arguments)
		);
	};
function $h(e, t) {
	var n = {};
	for (var r in e) Object.prototype.hasOwnProperty.call(e, r) && t.indexOf(r) < 0 && (n[r] = e[r]);
	if (e != null && typeof Object.getOwnPropertySymbols == 'function') for (var o = 0, r = Object.getOwnPropertySymbols(e); o < r.length; o++) t.indexOf(r[o]) < 0 && Object.prototype.propertyIsEnumerable.call(e, r[o]) && (n[r[o]] = e[r[o]]);
	return n;
}
function HS(e, t, n) {
	if (n || arguments.length === 2) for (var r = 0, o = t.length, i; r < o; r++) (i || !(r in t)) && (i || (i = Array.prototype.slice.call(t, 0, r)), (i[r] = t[r]));
	return e.concat(i || Array.prototype.slice.call(t));
}
var Oi = 'right-scroll-bar-position',
	ji = 'width-before-scroll-bar',
	WS = 'with-scroll-bars-hidden',
	VS = '--removed-body-scroll-bar-size';
function Ll(e, t) {
	return typeof e == 'function' ? e(t) : e && (e.current = t), e;
}
function GS(e, t) {
	var n = p.useState(function () {
		return {
			value: e,
			callback: t,
			facade: {
				get current() {
					return n.value;
				},
				set current(r) {
					var o = n.value;
					o !== r && ((n.value = r), n.callback(r, o));
				},
			},
		};
	})[0];
	return (n.callback = t), n.facade;
}
var KS = typeof window < 'u' ? p.useLayoutEffect : p.useEffect,
	lf = new WeakMap();
function YS(e, t) {
	var n = GS(null, function (r) {
		return e.forEach(function (o) {
			return Ll(o, r);
		});
	});
	return (
		KS(
			function () {
				var r = lf.get(n);
				if (r) {
					var o = new Set(r),
						i = new Set(e),
						s = n.current;
					o.forEach(function (l) {
						i.has(l) || Ll(l, null);
					}),
						i.forEach(function (l) {
							o.has(l) || Ll(l, s);
						});
				}
				lf.set(n, e);
			},
			[e]
		),
		n
	);
}
function QS(e) {
	return e;
}
function XS(e, t) {
	t === void 0 && (t = QS);
	var n = [],
		r = !1,
		o = {
			read: function () {
				if (r) throw new Error('Sidecar: could not `read` from an `assigned` medium. `read` could be used only with `useMedium`.');
				return n.length ? n[n.length - 1] : e;
			},
			useMedium: function (i) {
				var s = t(i, r);
				return (
					n.push(s),
					function () {
						n = n.filter(function (l) {
							return l !== s;
						});
					}
				);
			},
			assignSyncMedium: function (i) {
				for (r = !0; n.length; ) {
					var s = n;
					(n = []), s.forEach(i);
				}
				n = {
					push: function (l) {
						return i(l);
					},
					filter: function () {
						return n;
					},
				};
			},
			assignMedium: function (i) {
				r = !0;
				var s = [];
				if (n.length) {
					var l = n;
					(n = []), l.forEach(i), (s = n);
				}
				var a = function () {
						var f = s;
						(s = []), f.forEach(i);
					},
					u = function () {
						return Promise.resolve().then(a);
					};
				u(),
					(n = {
						push: function (f) {
							s.push(f), u();
						},
						filter: function (f) {
							return (s = s.filter(f)), n;
						},
					});
			},
		};
	return o;
}
function ZS(e) {
	e === void 0 && (e = {});
	var t = XS(null);
	return (t.options = kt({async: !0, ssr: !1}, e)), t;
}
var Bh = function (e) {
	var t = e.sideCar,
		n = $h(e, ['sideCar']);
	if (!t) throw new Error('Sidecar: please provide `sideCar` property to import the right car');
	var r = t.read();
	if (!r) throw new Error('Sidecar medium not found');
	return p.createElement(r, kt({}, n));
};
Bh.isSideCarExport = !0;
function qS(e, t) {
	return e.useMedium(t), Bh;
}
var Uh = ZS(),
	Fl = function () {},
	Vs = p.forwardRef(function (e, t) {
		var n = p.useRef(null),
			r = p.useState({onScrollCapture: Fl, onWheelCapture: Fl, onTouchMoveCapture: Fl}),
			o = r[0],
			i = r[1],
			s = e.forwardProps,
			l = e.children,
			a = e.className,
			u = e.removeScrollBar,
			f = e.enabled,
			d = e.shards,
			c = e.sideCar,
			g = e.noIsolation,
			w = e.inert,
			y = e.allowPinchZoom,
			S = e.as,
			m = S === void 0 ? 'div' : S,
			h = e.gapMode,
			x = $h(e, ['forwardProps', 'children', 'className', 'removeScrollBar', 'enabled', 'shards', 'sideCar', 'noIsolation', 'inert', 'allowPinchZoom', 'as', 'gapMode']),
			E = c,
			C = YS([n, t]),
			T = kt(kt({}, x), o);
		return p.createElement(
			p.Fragment,
			null,
			f && p.createElement(E, {sideCar: Uh, removeScrollBar: u, shards: d, noIsolation: g, inert: w, setCallbacks: i, allowPinchZoom: !!y, lockRef: n, gapMode: h}),
			s ? p.cloneElement(p.Children.only(l), kt(kt({}, T), {ref: C})) : p.createElement(m, kt({}, T, {className: a, ref: C}), l)
		);
	});
Vs.defaultProps = {enabled: !0, removeScrollBar: !0, inert: !1};
Vs.classNames = {fullWidth: ji, zeroRight: Oi};
var JS = function () {
	if (typeof __webpack_nonce__ < 'u') return __webpack_nonce__;
};
function eE() {
	if (!document) return null;
	var e = document.createElement('style');
	e.type = 'text/css';
	var t = JS();
	return t && e.setAttribute('nonce', t), e;
}
function tE(e, t) {
	e.styleSheet ? (e.styleSheet.cssText = t) : e.appendChild(document.createTextNode(t));
}
function nE(e) {
	var t = document.head || document.getElementsByTagName('head')[0];
	t.appendChild(e);
}
var rE = function () {
		var e = 0,
			t = null;
		return {
			add: function (n) {
				e == 0 && (t = eE()) && (tE(t, n), nE(t)), e++;
			},
			remove: function () {
				e--, !e && t && (t.parentNode && t.parentNode.removeChild(t), (t = null));
			},
		};
	},
	oE = function () {
		var e = rE();
		return function (t, n) {
			p.useEffect(
				function () {
					return (
						e.add(t),
						function () {
							e.remove();
						}
					);
				},
				[t && n]
			);
		};
	},
	Hh = function () {
		var e = oE(),
			t = function (n) {
				var r = n.styles,
					o = n.dynamic;
				return e(r, o), null;
			};
		return t;
	},
	iE = {left: 0, top: 0, right: 0, gap: 0},
	zl = function (e) {
		return parseInt(e || '', 10) || 0;
	},
	sE = function (e) {
		var t = window.getComputedStyle(document.body),
			n = t[e === 'padding' ? 'paddingLeft' : 'marginLeft'],
			r = t[e === 'padding' ? 'paddingTop' : 'marginTop'],
			o = t[e === 'padding' ? 'paddingRight' : 'marginRight'];
		return [zl(n), zl(r), zl(o)];
	},
	lE = function (e) {
		if ((e === void 0 && (e = 'margin'), typeof window > 'u')) return iE;
		var t = sE(e),
			n = document.documentElement.clientWidth,
			r = window.innerWidth;
		return {left: t[0], top: t[1], right: t[2], gap: Math.max(0, r - n + t[2] - t[0])};
	},
	aE = Hh(),
	wr = 'data-scroll-locked',
	uE = function (e, t, n, r) {
		var o = e.left,
			i = e.top,
			s = e.right,
			l = e.gap;
		return (
			n === void 0 && (n = 'margin'),
			`
  .`
				.concat(
					WS,
					` {
   overflow: hidden `
				)
				.concat(
					r,
					`;
   padding-right: `
				)
				.concat(l, 'px ')
				.concat(
					r,
					`;
  }
  body[`
				)
				.concat(
					wr,
					`] {
    overflow: hidden `
				)
				.concat(
					r,
					`;
    overscroll-behavior: contain;
    `
				)
				.concat(
					[
						t && 'position: relative '.concat(r, ';'),
						n === 'margin' &&
							`
    padding-left: `
								.concat(
									o,
									`px;
    padding-top: `
								)
								.concat(
									i,
									`px;
    padding-right: `
								)
								.concat(
									s,
									`px;
    margin-left:0;
    margin-top:0;
    margin-right: `
								)
								.concat(l, 'px ')
								.concat(
									r,
									`;
    `
								),
						n === 'padding' && 'padding-right: '.concat(l, 'px ').concat(r, ';'),
					]
						.filter(Boolean)
						.join(''),
					`
  }
  
  .`
				)
				.concat(
					Oi,
					` {
    right: `
				)
				.concat(l, 'px ')
				.concat(
					r,
					`;
  }
  
  .`
				)
				.concat(
					ji,
					` {
    margin-right: `
				)
				.concat(l, 'px ')
				.concat(
					r,
					`;
  }
  
  .`
				)
				.concat(Oi, ' .')
				.concat(
					Oi,
					` {
    right: 0 `
				)
				.concat(
					r,
					`;
  }
  
  .`
				)
				.concat(ji, ' .')
				.concat(
					ji,
					` {
    margin-right: 0 `
				)
				.concat(
					r,
					`;
  }
  
  body[`
				)
				.concat(
					wr,
					`] {
    `
				)
				.concat(VS, ': ')
				.concat(
					l,
					`px;
  }
`
				)
		);
	},
	af = function () {
		var e = parseInt(document.body.getAttribute(wr) || '0', 10);
		return isFinite(e) ? e : 0;
	},
	cE = function () {
		p.useEffect(function () {
			return (
				document.body.setAttribute(wr, (af() + 1).toString()),
				function () {
					var e = af() - 1;
					e <= 0 ? document.body.removeAttribute(wr) : document.body.setAttribute(wr, e.toString());
				}
			);
		}, []);
	},
	dE = function (e) {
		var t = e.noRelative,
			n = e.noImportant,
			r = e.gapMode,
			o = r === void 0 ? 'margin' : r;
		cE();
		var i = p.useMemo(
			function () {
				return lE(o);
			},
			[o]
		);
		return p.createElement(aE, {styles: uE(i, !t, o, n ? '' : '!important')});
	},
	Ga = !1;
if (typeof window < 'u')
	try {
		var yi = Object.defineProperty({}, 'passive', {
			get: function () {
				return (Ga = !0), !0;
			},
		});
		window.addEventListener('test', yi, yi), window.removeEventListener('test', yi, yi);
	} catch {
		Ga = !1;
	}
var Xn = Ga ? {passive: !1} : !1,
	fE = function (e) {
		return e.tagName === 'TEXTAREA';
	},
	Wh = function (e, t) {
		if (!(e instanceof Element)) return !1;
		var n = window.getComputedStyle(e);
		return n[t] !== 'hidden' && !(n.overflowY === n.overflowX && !fE(e) && n[t] === 'visible');
	},
	pE = function (e) {
		return Wh(e, 'overflowY');
	},
	mE = function (e) {
		return Wh(e, 'overflowX');
	},
	uf = function (e, t) {
		var n = t.ownerDocument,
			r = t;
		do {
			typeof ShadowRoot < 'u' && r instanceof ShadowRoot && (r = r.host);
			var o = Vh(e, r);
			if (o) {
				var i = Gh(e, r),
					s = i[1],
					l = i[2];
				if (s > l) return !0;
			}
			r = r.parentNode;
		} while (r && r !== n.body);
		return !1;
	},
	hE = function (e) {
		var t = e.scrollTop,
			n = e.scrollHeight,
			r = e.clientHeight;
		return [t, n, r];
	},
	gE = function (e) {
		var t = e.scrollLeft,
			n = e.scrollWidth,
			r = e.clientWidth;
		return [t, n, r];
	},
	Vh = function (e, t) {
		return e === 'v' ? pE(t) : mE(t);
	},
	Gh = function (e, t) {
		return e === 'v' ? hE(t) : gE(t);
	},
	vE = function (e, t) {
		return e === 'h' && t === 'rtl' ? -1 : 1;
	},
	yE = function (e, t, n, r, o) {
		var i = vE(e, window.getComputedStyle(t).direction),
			s = i * r,
			l = n.target,
			a = t.contains(l),
			u = !1,
			f = s > 0,
			d = 0,
			c = 0;
		do {
			var g = Gh(e, l),
				w = g[0],
				y = g[1],
				S = g[2],
				m = y - S - i * w;
			(w || m) && Vh(e, l) && ((d += m), (c += w)), l instanceof ShadowRoot ? (l = l.host) : (l = l.parentNode);
		} while ((!a && l !== document.body) || (a && (t.contains(l) || t === l)));
		return ((f && (Math.abs(d) < 1 || !o)) || (!f && (Math.abs(c) < 1 || !o))) && (u = !0), u;
	},
	wi = function (e) {
		return 'changedTouches' in e ? [e.changedTouches[0].clientX, e.changedTouches[0].clientY] : [0, 0];
	},
	cf = function (e) {
		return [e.deltaX, e.deltaY];
	},
	df = function (e) {
		return e && 'current' in e ? e.current : e;
	},
	wE = function (e, t) {
		return e[0] === t[0] && e[1] === t[1];
	},
	xE = function (e) {
		return `
  .block-interactivity-`
			.concat(
				e,
				` {pointer-events: none;}
  .allow-interactivity-`
			)
			.concat(
				e,
				` {pointer-events: all;}
`
			);
	},
	SE = 0,
	Zn = [];
function EE(e) {
	var t = p.useRef([]),
		n = p.useRef([0, 0]),
		r = p.useRef(),
		o = p.useState(SE++)[0],
		i = p.useState(Hh)[0],
		s = p.useRef(e);
	p.useEffect(
		function () {
			s.current = e;
		},
		[e]
	),
		p.useEffect(
			function () {
				if (e.inert) {
					document.body.classList.add('block-interactivity-'.concat(o));
					var y = HS([e.lockRef.current], (e.shards || []).map(df), !0).filter(Boolean);
					return (
						y.forEach(function (S) {
							return S.classList.add('allow-interactivity-'.concat(o));
						}),
						function () {
							document.body.classList.remove('block-interactivity-'.concat(o)),
								y.forEach(function (S) {
									return S.classList.remove('allow-interactivity-'.concat(o));
								});
						}
					);
				}
			},
			[e.inert, e.lockRef.current, e.shards]
		);
	var l = p.useCallback(function (y, S) {
			if (('touches' in y && y.touches.length === 2) || (y.type === 'wheel' && y.ctrlKey)) return !s.current.allowPinchZoom;
			var m = wi(y),
				h = n.current,
				x = 'deltaX' in y ? y.deltaX : h[0] - m[0],
				E = 'deltaY' in y ? y.deltaY : h[1] - m[1],
				C,
				T = y.target,
				k = Math.abs(x) > Math.abs(E) ? 'h' : 'v';
			if ('touches' in y && k === 'h' && T.type === 'range') return !1;
			var N = uf(k, T);
			if (!N) return !0;
			if ((N ? (C = k) : ((C = k === 'v' ? 'h' : 'v'), (N = uf(k, T))), !N)) return !1;
			if ((!r.current && 'changedTouches' in y && (x || E) && (r.current = C), !C)) return !0;
			var D = r.current || C;
			return yE(D, S, y, D === 'h' ? x : E, !0);
		}, []),
		a = p.useCallback(function (y) {
			var S = y;
			if (!(!Zn.length || Zn[Zn.length - 1] !== i)) {
				var m = 'deltaY' in S ? cf(S) : wi(S),
					h = t.current.filter(function (C) {
						return C.name === S.type && (C.target === S.target || S.target === C.shadowParent) && wE(C.delta, m);
					})[0];
				if (h && h.should) {
					S.cancelable && S.preventDefault();
					return;
				}
				if (!h) {
					var x = (s.current.shards || [])
							.map(df)
							.filter(Boolean)
							.filter(function (C) {
								return C.contains(S.target);
							}),
						E = x.length > 0 ? l(S, x[0]) : !s.current.noIsolation;
					E && S.cancelable && S.preventDefault();
				}
			}
		}, []),
		u = p.useCallback(function (y, S, m, h) {
			var x = {name: y, delta: S, target: m, should: h, shadowParent: CE(m)};
			t.current.push(x),
				setTimeout(function () {
					t.current = t.current.filter(function (E) {
						return E !== x;
					});
				}, 1);
		}, []),
		f = p.useCallback(function (y) {
			(n.current = wi(y)), (r.current = void 0);
		}, []),
		d = p.useCallback(function (y) {
			u(y.type, cf(y), y.target, l(y, e.lockRef.current));
		}, []),
		c = p.useCallback(function (y) {
			u(y.type, wi(y), y.target, l(y, e.lockRef.current));
		}, []);
	p.useEffect(function () {
		return (
			Zn.push(i),
			e.setCallbacks({onScrollCapture: d, onWheelCapture: d, onTouchMoveCapture: c}),
			document.addEventListener('wheel', a, Xn),
			document.addEventListener('touchmove', a, Xn),
			document.addEventListener('touchstart', f, Xn),
			function () {
				(Zn = Zn.filter(function (y) {
					return y !== i;
				})),
					document.removeEventListener('wheel', a, Xn),
					document.removeEventListener('touchmove', a, Xn),
					document.removeEventListener('touchstart', f, Xn);
			}
		);
	}, []);
	var g = e.removeScrollBar,
		w = e.inert;
	return p.createElement(p.Fragment, null, w ? p.createElement(i, {styles: xE(o)}) : null, g ? p.createElement(dE, {gapMode: e.gapMode}) : null);
}
function CE(e) {
	for (var t = null; e !== null; ) e instanceof ShadowRoot && ((t = e.host), (e = e.host)), (e = e.parentNode);
	return t;
}
const bE = qS(Uh, EE);
var ac = p.forwardRef(function (e, t) {
	return p.createElement(Vs, kt({}, e, {ref: t, sideCar: bE}));
});
ac.classNames = Vs.classNames;
var Ka = ['Enter', ' '],
	kE = ['ArrowDown', 'PageUp', 'Home'],
	Kh = ['ArrowUp', 'PageDown', 'End'],
	TE = [...kE, ...Kh],
	PE = {ltr: [...Ka, 'ArrowRight'], rtl: [...Ka, 'ArrowLeft']},
	NE = {ltr: ['ArrowLeft'], rtl: ['ArrowRight']},
	Ho = 'Menu',
	[Oo, _E, RE] = Nh(Ho),
	[Un, Yh] = Os(Ho, [RE, Fs, Oh]),
	Gs = Fs(),
	Qh = Oh(),
	[AE, Hn] = Un(Ho),
	[ME, Wo] = Un(Ho),
	Xh = (e) => {
		const {__scopeMenu: t, open: n = !1, children: r, dir: o, onOpenChange: i, modal: s = !0} = e,
			l = Gs(t),
			[a, u] = p.useState(null),
			f = p.useRef(!1),
			d = gt(i),
			c = _h(o);
		return (
			p.useEffect(() => {
				const g = () => {
						(f.current = !0), document.addEventListener('pointerdown', w, {capture: !0, once: !0}), document.addEventListener('pointermove', w, {capture: !0, once: !0});
					},
					w = () => (f.current = !1);
				return (
					document.addEventListener('keydown', g, {capture: !0}),
					() => {
						document.removeEventListener('keydown', g, {capture: !0}), document.removeEventListener('pointerdown', w, {capture: !0}), document.removeEventListener('pointermove', w, {capture: !0});
					}
				);
			}, []),
			v.jsx(fh, {...l, children: v.jsx(AE, {scope: t, open: n, onOpenChange: d, content: a, onContentChange: u, children: v.jsx(ME, {scope: t, onClose: p.useCallback(() => d(!1), [d]), isUsingKeyboardRef: f, dir: c, modal: s, children: r})})})
		);
	};
Xh.displayName = Ho;
var OE = 'MenuAnchor',
	uc = p.forwardRef((e, t) => {
		const {__scopeMenu: n, ...r} = e,
			o = Gs(n);
		return v.jsx(ph, {...o, ...r, ref: t});
	});
uc.displayName = OE;
var cc = 'MenuPortal',
	[jE, Zh] = Un(cc, {forceMount: void 0}),
	qh = (e) => {
		const {__scopeMenu: t, forceMount: n, children: r, container: o} = e,
			i = Hn(cc, t);
		return v.jsx(jE, {scope: t, forceMount: n, children: v.jsx(Wt, {present: n || i.open, children: v.jsx(rc, {asChild: !0, container: o, children: r})})});
	};
qh.displayName = cc;
var rt = 'MenuContent',
	[IE, dc] = Un(rt),
	Jh = p.forwardRef((e, t) => {
		const n = Zh(rt, e.__scopeMenu),
			{forceMount: r = n.forceMount, ...o} = e,
			i = Hn(rt, e.__scopeMenu),
			s = Wo(rt, e.__scopeMenu);
		return v.jsx(Oo.Provider, {scope: e.__scopeMenu, children: v.jsx(Wt, {present: r || i.open, children: v.jsx(Oo.Slot, {scope: e.__scopeMenu, children: s.modal ? v.jsx(DE, {...o, ref: t}) : v.jsx(LE, {...o, ref: t})})})});
	}),
	DE = p.forwardRef((e, t) => {
		const n = Hn(rt, e.__scopeMenu),
			r = p.useRef(null),
			o = we(t, r);
		return (
			p.useEffect(() => {
				const i = r.current;
				if (i) return zh(i);
			}, []),
			v.jsx(fc, {...e, ref: o, trapFocus: n.open, disableOutsidePointerEvents: n.open, disableOutsideScroll: !0, onFocusOutside: B(e.onFocusOutside, (i) => i.preventDefault(), {checkForDefaultPrevented: !1}), onDismiss: () => n.onOpenChange(!1)})
		);
	}),
	LE = p.forwardRef((e, t) => {
		const n = Hn(rt, e.__scopeMenu);
		return v.jsx(fc, {...e, ref: t, trapFocus: !1, disableOutsidePointerEvents: !1, disableOutsideScroll: !1, onDismiss: () => n.onOpenChange(!1)});
	}),
	fc = p.forwardRef((e, t) => {
		const {
				__scopeMenu: n,
				loop: r = !1,
				trapFocus: o,
				onOpenAutoFocus: i,
				onCloseAutoFocus: s,
				disableOutsidePointerEvents: l,
				onEntryFocus: a,
				onEscapeKeyDown: u,
				onPointerDownOutside: f,
				onFocusOutside: d,
				onInteractOutside: c,
				onDismiss: g,
				disableOutsideScroll: w,
				...y
			} = e,
			S = Hn(rt, n),
			m = Wo(rt, n),
			h = Gs(n),
			x = Qh(n),
			E = _E(n),
			[C, T] = p.useState(null),
			k = p.useRef(null),
			N = we(t, k, S.onContentChange),
			D = p.useRef(0),
			M = p.useRef(''),
			z = p.useRef(0),
			R = p.useRef(null),
			$ = p.useRef('right'),
			U = p.useRef(0),
			Q = w ? ac : p.Fragment,
			F = w ? {as: Dn, allowPinchZoom: !0} : void 0,
			X = (P) => {
				var xe, st;
				const j = M.current + P,
					O = E().filter((Se) => !Se.disabled),
					L = document.activeElement,
					Y = (xe = O.find((Se) => Se.ref.current === L)) == null ? void 0 : xe.textValue,
					ie = O.map((Se) => Se.textValue),
					Ne = QE(ie, j, Y),
					K = (st = O.find((Se) => Se.textValue === Ne)) == null ? void 0 : st.ref.current;
				(function Se(lt) {
					(M.current = lt), window.clearTimeout(D.current), lt !== '' && (D.current = window.setTimeout(() => Se(''), 1e3));
				})(j),
					K && setTimeout(() => K.focus());
			};
		p.useEffect(() => () => window.clearTimeout(D.current), []), Rh();
		const b = p.useCallback((P) => {
			var O, L;
			return $.current === ((O = R.current) == null ? void 0 : O.side) && ZE(P, (L = R.current) == null ? void 0 : L.area);
		}, []);
		return v.jsx(IE, {
			scope: n,
			searchRef: M,
			onItemEnter: p.useCallback(
				(P) => {
					b(P) && P.preventDefault();
				},
				[b]
			),
			onItemLeave: p.useCallback(
				(P) => {
					var j;
					b(P) || ((j = k.current) == null || j.focus(), T(null));
				},
				[b]
			),
			onTriggerLeave: p.useCallback(
				(P) => {
					b(P) && P.preventDefault();
				},
				[b]
			),
			pointerGraceTimerRef: z,
			onPointerGraceIntentChange: p.useCallback((P) => {
				R.current = P;
			}, []),
			children: v.jsx(Q, {
				...F,
				children: v.jsx(lc, {
					asChild: !0,
					trapped: o,
					onMountAutoFocus: B(i, (P) => {
						var j;
						P.preventDefault(), (j = k.current) == null || j.focus({preventScroll: !0});
					}),
					onUnmountAutoFocus: s,
					children: v.jsx(js, {
						asChild: !0,
						disableOutsidePointerEvents: l,
						onEscapeKeyDown: u,
						onPointerDownOutside: f,
						onFocusOutside: d,
						onInteractOutside: c,
						onDismiss: g,
						children: v.jsx(FS, {
							asChild: !0,
							...x,
							dir: m.dir,
							orientation: 'vertical',
							loop: r,
							currentTabStopId: C,
							onCurrentTabStopIdChange: T,
							onEntryFocus: B(a, (P) => {
								m.isUsingKeyboardRef.current || P.preventDefault();
							}),
							preventScrollOnEntryFocus: !0,
							children: v.jsx(mh, {
								role: 'menu',
								'aria-orientation': 'vertical',
								'data-state': hg(S.open),
								'data-radix-menu-content': '',
								dir: m.dir,
								...h,
								...y,
								ref: N,
								style: {outline: 'none', ...y.style},
								onKeyDown: B(y.onKeyDown, (P) => {
									const O = P.target.closest('[data-radix-menu-content]') === P.currentTarget,
										L = P.ctrlKey || P.altKey || P.metaKey,
										Y = P.key.length === 1;
									O && (P.key === 'Tab' && P.preventDefault(), !L && Y && X(P.key));
									const ie = k.current;
									if (P.target !== ie || !TE.includes(P.key)) return;
									P.preventDefault();
									const K = E()
										.filter((xe) => !xe.disabled)
										.map((xe) => xe.ref.current);
									Kh.includes(P.key) && K.reverse(), KE(K);
								}),
								onBlur: B(e.onBlur, (P) => {
									P.currentTarget.contains(P.target) || (window.clearTimeout(D.current), (M.current = ''));
								}),
								onPointerMove: B(
									e.onPointerMove,
									jo((P) => {
										const j = P.target,
											O = U.current !== P.clientX;
										if (P.currentTarget.contains(j) && O) {
											const L = P.clientX > U.current ? 'right' : 'left';
											($.current = L), (U.current = P.clientX);
										}
									})
								),
							}),
						}),
					}),
				}),
			}),
		});
	});
Jh.displayName = rt;
var FE = 'MenuGroup',
	pc = p.forwardRef((e, t) => {
		const {__scopeMenu: n, ...r} = e;
		return v.jsx(pe.div, {role: 'group', ...r, ref: t});
	});
pc.displayName = FE;
var zE = 'MenuLabel',
	eg = p.forwardRef((e, t) => {
		const {__scopeMenu: n, ...r} = e;
		return v.jsx(pe.div, {...r, ref: t});
	});
eg.displayName = zE;
var ds = 'MenuItem',
	ff = 'menu.itemSelect',
	Ks = p.forwardRef((e, t) => {
		const {disabled: n = !1, onSelect: r, ...o} = e,
			i = p.useRef(null),
			s = Wo(ds, e.__scopeMenu),
			l = dc(ds, e.__scopeMenu),
			a = we(t, i),
			u = p.useRef(!1),
			f = () => {
				const d = i.current;
				if (!n && d) {
					const c = new CustomEvent(ff, {bubbles: !0, cancelable: !0});
					d.addEventListener(ff, (g) => (r == null ? void 0 : r(g)), {once: !0}), Vm(d, c), c.defaultPrevented ? (u.current = !1) : s.onClose();
				}
			};
		return v.jsx(tg, {
			...o,
			ref: a,
			disabled: n,
			onClick: B(e.onClick, f),
			onPointerDown: (d) => {
				var c;
				(c = e.onPointerDown) == null || c.call(e, d), (u.current = !0);
			},
			onPointerUp: B(e.onPointerUp, (d) => {
				var c;
				u.current || (c = d.currentTarget) == null || c.click();
			}),
			onKeyDown: B(e.onKeyDown, (d) => {
				const c = l.searchRef.current !== '';
				n || (c && d.key === ' ') || (Ka.includes(d.key) && (d.currentTarget.click(), d.preventDefault()));
			}),
		});
	});
Ks.displayName = ds;
var tg = p.forwardRef((e, t) => {
		const {__scopeMenu: n, disabled: r = !1, textValue: o, ...i} = e,
			s = dc(ds, n),
			l = Qh(n),
			a = p.useRef(null),
			u = we(t, a),
			[f, d] = p.useState(!1),
			[c, g] = p.useState('');
		return (
			p.useEffect(() => {
				const w = a.current;
				w && g((w.textContent ?? '').trim());
			}, [i.children]),
			v.jsx(Oo.ItemSlot, {
				scope: n,
				disabled: r,
				textValue: o ?? c,
				children: v.jsx(zS, {
					asChild: !0,
					...l,
					focusable: !r,
					children: v.jsx(pe.div, {
						role: 'menuitem',
						'data-highlighted': f ? '' : void 0,
						'aria-disabled': r || void 0,
						'data-disabled': r ? '' : void 0,
						...i,
						ref: u,
						onPointerMove: B(
							e.onPointerMove,
							jo((w) => {
								r ? s.onItemLeave(w) : (s.onItemEnter(w), w.defaultPrevented || w.currentTarget.focus({preventScroll: !0}));
							})
						),
						onPointerLeave: B(
							e.onPointerLeave,
							jo((w) => s.onItemLeave(w))
						),
						onFocus: B(e.onFocus, () => d(!0)),
						onBlur: B(e.onBlur, () => d(!1)),
					}),
				}),
			})
		);
	}),
	$E = 'MenuCheckboxItem',
	ng = p.forwardRef((e, t) => {
		const {checked: n = !1, onCheckedChange: r, ...o} = e;
		return v.jsx(lg, {
			scope: e.__scopeMenu,
			checked: n,
			children: v.jsx(Ks, {role: 'menuitemcheckbox', 'aria-checked': fs(n) ? 'mixed' : n, ...o, ref: t, 'data-state': hc(n), onSelect: B(o.onSelect, () => (r == null ? void 0 : r(fs(n) ? !0 : !n)), {checkForDefaultPrevented: !1})}),
		});
	});
ng.displayName = $E;
var rg = 'MenuRadioGroup',
	[BE, UE] = Un(rg, {value: void 0, onValueChange: () => {}}),
	og = p.forwardRef((e, t) => {
		const {value: n, onValueChange: r, ...o} = e,
			i = gt(r);
		return v.jsx(BE, {scope: e.__scopeMenu, value: n, onValueChange: i, children: v.jsx(pc, {...o, ref: t})});
	});
og.displayName = rg;
var ig = 'MenuRadioItem',
	sg = p.forwardRef((e, t) => {
		const {value: n, ...r} = e,
			o = UE(ig, e.__scopeMenu),
			i = n === o.value;
		return v.jsx(lg, {
			scope: e.__scopeMenu,
			checked: i,
			children: v.jsx(Ks, {
				role: 'menuitemradio',
				'aria-checked': i,
				...r,
				ref: t,
				'data-state': hc(i),
				onSelect: B(
					r.onSelect,
					() => {
						var s;
						return (s = o.onValueChange) == null ? void 0 : s.call(o, n);
					},
					{checkForDefaultPrevented: !1}
				),
			}),
		});
	});
sg.displayName = ig;
var mc = 'MenuItemIndicator',
	[lg, HE] = Un(mc, {checked: !1}),
	ag = p.forwardRef((e, t) => {
		const {__scopeMenu: n, forceMount: r, ...o} = e,
			i = HE(mc, n);
		return v.jsx(Wt, {present: r || fs(i.checked) || i.checked === !0, children: v.jsx(pe.span, {...o, ref: t, 'data-state': hc(i.checked)})});
	});
ag.displayName = mc;
var WE = 'MenuSeparator',
	ug = p.forwardRef((e, t) => {
		const {__scopeMenu: n, ...r} = e;
		return v.jsx(pe.div, {role: 'separator', 'aria-orientation': 'horizontal', ...r, ref: t});
	});
ug.displayName = WE;
var VE = 'MenuArrow',
	cg = p.forwardRef((e, t) => {
		const {__scopeMenu: n, ...r} = e,
			o = Gs(n);
		return v.jsx(hh, {...o, ...r, ref: t});
	});
cg.displayName = VE;
var GE = 'MenuSub',
	[bk, dg] = Un(GE),
	no = 'MenuSubTrigger',
	fg = p.forwardRef((e, t) => {
		const n = Hn(no, e.__scopeMenu),
			r = Wo(no, e.__scopeMenu),
			o = dg(no, e.__scopeMenu),
			i = dc(no, e.__scopeMenu),
			s = p.useRef(null),
			{pointerGraceTimerRef: l, onPointerGraceIntentChange: a} = i,
			u = {__scopeMenu: e.__scopeMenu},
			f = p.useCallback(() => {
				s.current && window.clearTimeout(s.current), (s.current = null);
			}, []);
		return (
			p.useEffect(() => f, [f]),
			p.useEffect(() => {
				const d = l.current;
				return () => {
					window.clearTimeout(d), a(null);
				};
			}, [l, a]),
			v.jsx(uc, {
				asChild: !0,
				...u,
				children: v.jsx(tg, {
					id: o.triggerId,
					'aria-haspopup': 'menu',
					'aria-expanded': n.open,
					'aria-controls': o.contentId,
					'data-state': hg(n.open),
					...e,
					ref: Ms(t, o.onTriggerChange),
					onClick: (d) => {
						var c;
						(c = e.onClick) == null || c.call(e, d), !(e.disabled || d.defaultPrevented) && (d.currentTarget.focus(), n.open || n.onOpenChange(!0));
					},
					onPointerMove: B(
						e.onPointerMove,
						jo((d) => {
							i.onItemEnter(d),
								!d.defaultPrevented &&
									!e.disabled &&
									!n.open &&
									!s.current &&
									(i.onPointerGraceIntentChange(null),
									(s.current = window.setTimeout(() => {
										n.onOpenChange(!0), f();
									}, 100)));
						})
					),
					onPointerLeave: B(
						e.onPointerLeave,
						jo((d) => {
							var g, w;
							f();
							const c = (g = n.content) == null ? void 0 : g.getBoundingClientRect();
							if (c) {
								const y = (w = n.content) == null ? void 0 : w.dataset.side,
									S = y === 'right',
									m = S ? -5 : 5,
									h = c[S ? 'left' : 'right'],
									x = c[S ? 'right' : 'left'];
								i.onPointerGraceIntentChange({
									area: [
										{x: d.clientX + m, y: d.clientY},
										{x: h, y: c.top},
										{x, y: c.top},
										{x, y: c.bottom},
										{x: h, y: c.bottom},
									],
									side: y,
								}),
									window.clearTimeout(l.current),
									(l.current = window.setTimeout(() => i.onPointerGraceIntentChange(null), 300));
							} else {
								if ((i.onTriggerLeave(d), d.defaultPrevented)) return;
								i.onPointerGraceIntentChange(null);
							}
						})
					),
					onKeyDown: B(e.onKeyDown, (d) => {
						var g;
						const c = i.searchRef.current !== '';
						e.disabled || (c && d.key === ' ') || (PE[r.dir].includes(d.key) && (n.onOpenChange(!0), (g = n.content) == null || g.focus(), d.preventDefault()));
					}),
				}),
			})
		);
	});
fg.displayName = no;
var pg = 'MenuSubContent',
	mg = p.forwardRef((e, t) => {
		const n = Zh(rt, e.__scopeMenu),
			{forceMount: r = n.forceMount, ...o} = e,
			i = Hn(rt, e.__scopeMenu),
			s = Wo(rt, e.__scopeMenu),
			l = dg(pg, e.__scopeMenu),
			a = p.useRef(null),
			u = we(t, a);
		return v.jsx(Oo.Provider, {
			scope: e.__scopeMenu,
			children: v.jsx(Wt, {
				present: r || i.open,
				children: v.jsx(Oo.Slot, {
					scope: e.__scopeMenu,
					children: v.jsx(fc, {
						id: l.contentId,
						'aria-labelledby': l.triggerId,
						...o,
						ref: u,
						align: 'start',
						side: s.dir === 'rtl' ? 'left' : 'right',
						disableOutsidePointerEvents: !1,
						disableOutsideScroll: !1,
						trapFocus: !1,
						onOpenAutoFocus: (f) => {
							var d;
							s.isUsingKeyboardRef.current && ((d = a.current) == null || d.focus()), f.preventDefault();
						},
						onCloseAutoFocus: (f) => f.preventDefault(),
						onFocusOutside: B(e.onFocusOutside, (f) => {
							f.target !== l.trigger && i.onOpenChange(!1);
						}),
						onEscapeKeyDown: B(e.onEscapeKeyDown, (f) => {
							s.onClose(), f.preventDefault();
						}),
						onKeyDown: B(e.onKeyDown, (f) => {
							var g;
							const d = f.currentTarget.contains(f.target),
								c = NE[s.dir].includes(f.key);
							d && c && (i.onOpenChange(!1), (g = l.trigger) == null || g.focus(), f.preventDefault());
						}),
					}),
				}),
			}),
		});
	});
mg.displayName = pg;
function hg(e) {
	return e ? 'open' : 'closed';
}
function fs(e) {
	return e === 'indeterminate';
}
function hc(e) {
	return fs(e) ? 'indeterminate' : e ? 'checked' : 'unchecked';
}
function KE(e) {
	const t = document.activeElement;
	for (const n of e) if (n === t || (n.focus(), document.activeElement !== t)) return;
}
function YE(e, t) {
	return e.map((n, r) => e[(t + r) % e.length]);
}
function QE(e, t, n) {
	const o = t.length > 1 && Array.from(t).every((u) => u === t[0]) ? t[0] : t,
		i = n ? e.indexOf(n) : -1;
	let s = YE(e, Math.max(i, 0));
	o.length === 1 && (s = s.filter((u) => u !== n));
	const a = s.find((u) => u.toLowerCase().startsWith(o.toLowerCase()));
	return a !== n ? a : void 0;
}
function XE(e, t) {
	const {x: n, y: r} = e;
	let o = !1;
	for (let i = 0, s = t.length - 1; i < t.length; s = i++) {
		const l = t[i].x,
			a = t[i].y,
			u = t[s].x,
			f = t[s].y;
		a > r != f > r && n < ((u - l) * (r - a)) / (f - a) + l && (o = !o);
	}
	return o;
}
function ZE(e, t) {
	if (!t) return !1;
	const n = {x: e.clientX, y: e.clientY};
	return XE(n, t);
}
function jo(e) {
	return (t) => (t.pointerType === 'mouse' ? e(t) : void 0);
}
var qE = Xh,
	JE = uc,
	eC = qh,
	tC = Jh,
	nC = pc,
	rC = eg,
	oC = Ks,
	iC = ng,
	sC = og,
	lC = sg,
	aC = ag,
	uC = ug,
	cC = cg,
	dC = fg,
	fC = mg,
	gc = 'DropdownMenu',
	[pC, kk] = Os(gc, [Yh]),
	Le = Yh(),
	[mC, gg] = pC(gc),
	vg = (e) => {
		const {__scopeDropdownMenu: t, children: n, dir: r, open: o, defaultOpen: i, onOpenChange: s, modal: l = !0} = e,
			a = Le(t),
			u = p.useRef(null),
			[f = !1, d] = zs({prop: o, defaultProp: i, onChange: s});
		return v.jsx(mC, {
			scope: t,
			triggerId: _n(),
			triggerRef: u,
			contentId: _n(),
			open: f,
			onOpenChange: d,
			onOpenToggle: p.useCallback(() => d((c) => !c), [d]),
			modal: l,
			children: v.jsx(qE, {...a, open: f, onOpenChange: d, dir: r, modal: l, children: n}),
		});
	};
vg.displayName = gc;
var yg = 'DropdownMenuTrigger',
	wg = p.forwardRef((e, t) => {
		const {__scopeDropdownMenu: n, disabled: r = !1, ...o} = e,
			i = gg(yg, n),
			s = Le(n);
		return v.jsx(JE, {
			asChild: !0,
			...s,
			children: v.jsx(pe.button, {
				type: 'button',
				id: i.triggerId,
				'aria-haspopup': 'menu',
				'aria-expanded': i.open,
				'aria-controls': i.open ? i.contentId : void 0,
				'data-state': i.open ? 'open' : 'closed',
				'data-disabled': r ? '' : void 0,
				disabled: r,
				...o,
				ref: Ms(t, i.triggerRef),
				onPointerDown: B(e.onPointerDown, (l) => {
					!r && l.button === 0 && l.ctrlKey === !1 && (i.onOpenToggle(), i.open || l.preventDefault());
				}),
				onKeyDown: B(e.onKeyDown, (l) => {
					r || (['Enter', ' '].includes(l.key) && i.onOpenToggle(), l.key === 'ArrowDown' && i.onOpenChange(!0), ['Enter', ' ', 'ArrowDown'].includes(l.key) && l.preventDefault());
				}),
			}),
		});
	});
wg.displayName = yg;
var hC = 'DropdownMenuPortal',
	xg = (e) => {
		const {__scopeDropdownMenu: t, ...n} = e,
			r = Le(t);
		return v.jsx(eC, {...r, ...n});
	};
xg.displayName = hC;
var Sg = 'DropdownMenuContent',
	Eg = p.forwardRef((e, t) => {
		const {__scopeDropdownMenu: n, ...r} = e,
			o = gg(Sg, n),
			i = Le(n),
			s = p.useRef(!1);
		return v.jsx(tC, {
			id: o.contentId,
			'aria-labelledby': o.triggerId,
			...i,
			...r,
			ref: t,
			onCloseAutoFocus: B(e.onCloseAutoFocus, (l) => {
				var a;
				s.current || (a = o.triggerRef.current) == null || a.focus(), (s.current = !1), l.preventDefault();
			}),
			onInteractOutside: B(e.onInteractOutside, (l) => {
				const a = l.detail.originalEvent,
					u = a.button === 0 && a.ctrlKey === !0,
					f = a.button === 2 || u;
				(!o.modal || f) && (s.current = !0);
			}),
			style: {
				...e.style,
				'--radix-dropdown-menu-content-transform-origin': 'var(--radix-popper-transform-origin)',
				'--radix-dropdown-menu-content-available-width': 'var(--radix-popper-available-width)',
				'--radix-dropdown-menu-content-available-height': 'var(--radix-popper-available-height)',
				'--radix-dropdown-menu-trigger-width': 'var(--radix-popper-anchor-width)',
				'--radix-dropdown-menu-trigger-height': 'var(--radix-popper-anchor-height)',
			},
		});
	});
Eg.displayName = Sg;
var gC = 'DropdownMenuGroup',
	vC = p.forwardRef((e, t) => {
		const {__scopeDropdownMenu: n, ...r} = e,
			o = Le(n);
		return v.jsx(nC, {...o, ...r, ref: t});
	});
vC.displayName = gC;
var yC = 'DropdownMenuLabel',
	Cg = p.forwardRef((e, t) => {
		const {__scopeDropdownMenu: n, ...r} = e,
			o = Le(n);
		return v.jsx(rC, {...o, ...r, ref: t});
	});
Cg.displayName = yC;
var wC = 'DropdownMenuItem',
	bg = p.forwardRef((e, t) => {
		const {__scopeDropdownMenu: n, ...r} = e,
			o = Le(n);
		return v.jsx(oC, {...o, ...r, ref: t});
	});
bg.displayName = wC;
var xC = 'DropdownMenuCheckboxItem',
	kg = p.forwardRef((e, t) => {
		const {__scopeDropdownMenu: n, ...r} = e,
			o = Le(n);
		return v.jsx(iC, {...o, ...r, ref: t});
	});
kg.displayName = xC;
var SC = 'DropdownMenuRadioGroup',
	EC = p.forwardRef((e, t) => {
		const {__scopeDropdownMenu: n, ...r} = e,
			o = Le(n);
		return v.jsx(sC, {...o, ...r, ref: t});
	});
EC.displayName = SC;
var CC = 'DropdownMenuRadioItem',
	Tg = p.forwardRef((e, t) => {
		const {__scopeDropdownMenu: n, ...r} = e,
			o = Le(n);
		return v.jsx(lC, {...o, ...r, ref: t});
	});
Tg.displayName = CC;
var bC = 'DropdownMenuItemIndicator',
	Pg = p.forwardRef((e, t) => {
		const {__scopeDropdownMenu: n, ...r} = e,
			o = Le(n);
		return v.jsx(aC, {...o, ...r, ref: t});
	});
Pg.displayName = bC;
var kC = 'DropdownMenuSeparator',
	Ng = p.forwardRef((e, t) => {
		const {__scopeDropdownMenu: n, ...r} = e,
			o = Le(n);
		return v.jsx(uC, {...o, ...r, ref: t});
	});
Ng.displayName = kC;
var TC = 'DropdownMenuArrow',
	PC = p.forwardRef((e, t) => {
		const {__scopeDropdownMenu: n, ...r} = e,
			o = Le(n);
		return v.jsx(cC, {...o, ...r, ref: t});
	});
PC.displayName = TC;
var NC = 'DropdownMenuSubTrigger',
	_g = p.forwardRef((e, t) => {
		const {__scopeDropdownMenu: n, ...r} = e,
			o = Le(n);
		return v.jsx(dC, {...o, ...r, ref: t});
	});
_g.displayName = NC;
var _C = 'DropdownMenuSubContent',
	Rg = p.forwardRef((e, t) => {
		const {__scopeDropdownMenu: n, ...r} = e,
			o = Le(n);
		return v.jsx(fC, {
			...o,
			...r,
			ref: t,
			style: {
				...e.style,
				'--radix-dropdown-menu-content-transform-origin': 'var(--radix-popper-transform-origin)',
				'--radix-dropdown-menu-content-available-width': 'var(--radix-popper-available-width)',
				'--radix-dropdown-menu-content-available-height': 'var(--radix-popper-available-height)',
				'--radix-dropdown-menu-trigger-width': 'var(--radix-popper-anchor-width)',
				'--radix-dropdown-menu-trigger-height': 'var(--radix-popper-anchor-height)',
			},
		});
	});
Rg.displayName = _C;
var RC = vg,
	AC = wg,
	MC = xg,
	Ag = Eg,
	Mg = Cg,
	Og = bg,
	jg = kg,
	Ig = Tg,
	Dg = Pg,
	Lg = Ng,
	Fg = _g,
	zg = Rg;
/**
 * @license lucide-react v0.453.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */ const OC = (e) => e.replace(/([a-z0-9])([A-Z])/g, '$1-$2').toLowerCase(),
	$g = (...e) => e.filter((t, n, r) => !!t && r.indexOf(t) === n).join(' ');
/**
 * @license lucide-react v0.453.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */ var jC = {xmlns: 'http://www.w3.org/2000/svg', width: 24, height: 24, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round', strokeLinejoin: 'round'};
/**
 * @license lucide-react v0.453.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */ const IC = p.forwardRef(({color: e = 'currentColor', size: t = 24, strokeWidth: n = 2, absoluteStrokeWidth: r, className: o = '', children: i, iconNode: s, ...l}, a) =>
	p.createElement('svg', {ref: a, ...jC, width: t, height: t, stroke: e, strokeWidth: r ? (Number(n) * 24) / Number(t) : n, className: $g('lucide', o), ...l}, [...s.map(([u, f]) => p.createElement(u, f)), ...(Array.isArray(i) ? i : [i])])
);
/**
 * @license lucide-react v0.453.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */ const Vo = (e, t) => {
	const n = p.forwardRef(({className: r, ...o}, i) => p.createElement(IC, {ref: i, iconNode: t, className: $g(`lucide-${OC(e)}`, r), ...o}));
	return (n.displayName = `${e}`), n;
};
/**
 * @license lucide-react v0.453.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */ const DC = Vo('Check', [['path', {d: 'M20 6 9 17l-5-5', key: '1gmf2c'}]]);
/**
 * @license lucide-react v0.453.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */ const LC = Vo('ChevronRight', [['path', {d: 'm9 18 6-6-6-6', key: 'mthhwq'}]]);
/**
 * @license lucide-react v0.453.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */ const FC = Vo('Circle', [['circle', {cx: '12', cy: '12', r: '10', key: '1mglay'}]]);
/**
 * @license lucide-react v0.453.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */ const zC = Vo('Menu', [
	['line', {x1: '4', x2: '20', y1: '12', y2: '12', key: '1e0a9i'}],
	['line', {x1: '4', x2: '20', y1: '6', y2: '6', key: '1owob3'}],
	['line', {x1: '4', x2: '20', y1: '18', y2: '18', key: 'yk5zj1'}],
]);
/**
 * @license lucide-react v0.453.0 - ISC
 *
 * This source code is licensed under the ISC license.
 * See the LICENSE file in the root directory of this source tree.
 */ const Bg = Vo('X', [
		['path', {d: 'M18 6 6 18', key: '1bl5f8'}],
		['path', {d: 'm6 6 12 12', key: 'd8bk6v'}],
	]),
	$C = RC,
	BC = AC,
	UC = p.forwardRef(({className: e, inset: t, children: n, ...r}, o) =>
		v.jsxs(Fg, {
			ref: o,
			className: he('flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none focus:bg-accent data-[state=open]:bg-accent', t && 'pl-8', e),
			...r,
			children: [n, v.jsx(LC, {className: 'ml-auto h-4 w-4'})],
		})
	);
UC.displayName = Fg.displayName;
const HC = p.forwardRef(({className: e, ...t}, n) =>
	v.jsx(zg, {
		ref: n,
		className: he(
			'z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-lg data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2',
			e
		),
		...t,
	})
);
HC.displayName = zg.displayName;
const Ug = p.forwardRef(({className: e, sideOffset: t = 4, ...n}, r) =>
	v.jsx(MC, {
		children: v.jsx(Ag, {
			ref: r,
			sideOffset: t,
			className: he(
				'z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2',
				e
			),
			...n,
		}),
	})
);
Ug.displayName = Ag.displayName;
const Hg = p.forwardRef(({className: e, inset: t, ...n}, r) =>
	v.jsx(Og, {
		ref: r,
		className: he(
			'relative flex cursor-default select-none items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none transition-colors focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0',
			t && 'pl-8',
			e
		),
		...n,
	})
);
Hg.displayName = Og.displayName;
const WC = p.forwardRef(({className: e, children: t, checked: n, ...r}, o) =>
	v.jsxs(jg, {
		ref: o,
		className: he(
			'relative flex cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none transition-colors focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
			e
		),
		checked: n,
		...r,
		children: [v.jsx('span', {className: 'absolute left-2 flex h-3.5 w-3.5 items-center justify-center', children: v.jsx(Dg, {children: v.jsx(DC, {className: 'h-4 w-4'})})}), t],
	})
);
WC.displayName = jg.displayName;
const VC = p.forwardRef(({className: e, children: t, ...n}, r) =>
	v.jsxs(Ig, {
		ref: r,
		className: he(
			'relative flex cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none transition-colors focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
			e
		),
		...n,
		children: [v.jsx('span', {className: 'absolute left-2 flex h-3.5 w-3.5 items-center justify-center', children: v.jsx(Dg, {children: v.jsx(FC, {className: 'h-2 w-2 fill-current'})})}), t],
	})
);
VC.displayName = Ig.displayName;
const GC = p.forwardRef(({className: e, inset: t, ...n}, r) => v.jsx(Mg, {ref: r, className: he('px-2 py-1.5 text-sm font-semibold', t && 'pl-8', e), ...n}));
GC.displayName = Mg.displayName;
const KC = p.forwardRef(({className: e, ...t}, n) => v.jsx(Lg, {ref: n, className: he('-mx-1 my-1 h-px bg-muted', e), ...t}));
KC.displayName = Lg.displayName;
const YC = (e) => {
		e = new Date(e);
		const t = {year: 'numeric', month: 'long', day: 'numeric'};
		return e.toLocaleDateString('en-US', t);
	},
	QC = (e) => {
		e = new Date(e);
		const t = {hour: '2-digit', minute: '2-digit', hour12: !1};
		return e.toLocaleTimeString('en-US', t);
	},
	XC = '_editSession_1nfqv_1',
	ZC = {editSession: XC},
	pf = ['#B4E64A', '#FFB800', '#B965CC', '#87DAC6', '#FF68C3'],
	mf = (e) => {
		const t = [...e].reduce((n, r) => n + r.charCodeAt(0), 0);
		return pf[t % pf.length];
	},
	Ya = ({agent: e, customer: t, tooltip: n = !0}) => {
		var d, c;
		const r = mf(e.id),
			o = t && mf(t.id),
			i = (e == null ? void 0 : e.name) === 'N/A',
			s = (t == null ? void 0 : t.name) === 'N/A',
			l = e.name === '<guest>' ? 'G' : e.name[0].toUpperCase(),
			a = (t == null ? void 0 : t.name) === '<guest>',
			u = a ? 'G' : (c = (d = t == null ? void 0 : t.name) == null ? void 0 : d[0]) == null ? void 0 : c.toUpperCase(),
			f = {transform: 'translateY(17px)', fontSize: '13px !important', fontWeight: 400, fontFamily: 'inter'};
		return (
			n || (f.display = 'none'),
			v.jsx(cs, {
				value: `${e.name} / ${!(t != null && t.name) || a ? 'Guest' : t.name}`,
				side: 'right',
				style: f,
				children: v.jsxs('div', {
					className: 'relative',
					children: [
						v.jsx('div', {
							style: {background: r},
							'aria-label': 'agent ' + e.name,
							className: Ia('me-[10px] size-[38px] rounded-full flex items-center justify-center text-white text-[20px] font-semibold', i && 'text-[14px] !bg-gray-300'),
							children: i ? 'N/A' : l,
						}),
						t &&
							v.jsx('div', {
								style: {background: o},
								'aria-label': 'customer ' + t.name,
								className: Ia('absolute me-[3px] size-[20px] rounded-full flex items-center justify-center text-white text-[12px] font-semibold border-[2px] border-white bottom-0 right-0 z-10', s && 'text-[8px] !bg-gray-300'),
								children: s ? 'N/A' : u,
							}),
					],
				}),
			})
		);
	},
	Rr = (e) => {
		(e.key === 'Enter' || e.key === ' ') && e.target.click();
	};
var vc = 'Dialog',
	[Wg, Tk] = Os(vc),
	[qC, wt] = Wg(vc),
	Vg = (e) => {
		const {__scopeDialog: t, children: n, open: r, defaultOpen: o, onOpenChange: i, modal: s = !0} = e,
			l = p.useRef(null),
			a = p.useRef(null),
			[u = !1, f] = zs({prop: r, defaultProp: o, onChange: i});
		return v.jsx(qC, {scope: t, triggerRef: l, contentRef: a, contentId: _n(), titleId: _n(), descriptionId: _n(), open: u, onOpenChange: f, onOpenToggle: p.useCallback(() => f((d) => !d), [f]), modal: s, children: n});
	};
Vg.displayName = vc;
var Gg = 'DialogTrigger',
	Kg = p.forwardRef((e, t) => {
		const {__scopeDialog: n, ...r} = e,
			o = wt(Gg, n),
			i = we(t, o.triggerRef);
		return v.jsx(pe.button, {type: 'button', 'aria-haspopup': 'dialog', 'aria-expanded': o.open, 'aria-controls': o.contentId, 'data-state': Ec(o.open), ...r, ref: i, onClick: B(e.onClick, o.onOpenToggle)});
	});
Kg.displayName = Gg;
var yc = 'DialogPortal',
	[JC, Yg] = Wg(yc, {forceMount: void 0}),
	Qg = (e) => {
		const {__scopeDialog: t, forceMount: n, children: r, container: o} = e,
			i = wt(yc, t);
		return v.jsx(JC, {scope: t, forceMount: n, children: p.Children.map(r, (s) => v.jsx(Wt, {present: n || i.open, children: v.jsx(rc, {asChild: !0, container: o, children: s})}))});
	};
Qg.displayName = yc;
var ps = 'DialogOverlay',
	Xg = p.forwardRef((e, t) => {
		const n = Yg(ps, e.__scopeDialog),
			{forceMount: r = n.forceMount, ...o} = e,
			i = wt(ps, e.__scopeDialog);
		return i.modal ? v.jsx(Wt, {present: r || i.open, children: v.jsx(eb, {...o, ref: t})}) : null;
	});
Xg.displayName = ps;
var eb = p.forwardRef((e, t) => {
		const {__scopeDialog: n, ...r} = e,
			o = wt(ps, n);
		return v.jsx(ac, {as: Dn, allowPinchZoom: !0, shards: [o.contentRef], children: v.jsx(pe.div, {'data-state': Ec(o.open), ...r, ref: t, style: {pointerEvents: 'auto', ...r.style}})});
	}),
	zn = 'DialogContent',
	Zg = p.forwardRef((e, t) => {
		const n = Yg(zn, e.__scopeDialog),
			{forceMount: r = n.forceMount, ...o} = e,
			i = wt(zn, e.__scopeDialog);
		return v.jsx(Wt, {present: r || i.open, children: i.modal ? v.jsx(tb, {...o, ref: t}) : v.jsx(nb, {...o, ref: t})});
	});
Zg.displayName = zn;
var tb = p.forwardRef((e, t) => {
		const n = wt(zn, e.__scopeDialog),
			r = p.useRef(null),
			o = we(t, n.contentRef, r);
		return (
			p.useEffect(() => {
				const i = r.current;
				if (i) return zh(i);
			}, []),
			v.jsx(qg, {
				...e,
				ref: o,
				trapFocus: n.open,
				disableOutsidePointerEvents: !0,
				onCloseAutoFocus: B(e.onCloseAutoFocus, (i) => {
					var s;
					i.preventDefault(), (s = n.triggerRef.current) == null || s.focus();
				}),
				onPointerDownOutside: B(e.onPointerDownOutside, (i) => {
					const s = i.detail.originalEvent,
						l = s.button === 0 && s.ctrlKey === !0;
					(s.button === 2 || l) && i.preventDefault();
				}),
				onFocusOutside: B(e.onFocusOutside, (i) => i.preventDefault()),
			})
		);
	}),
	nb = p.forwardRef((e, t) => {
		const n = wt(zn, e.__scopeDialog),
			r = p.useRef(!1),
			o = p.useRef(!1);
		return v.jsx(qg, {
			...e,
			ref: t,
			trapFocus: !1,
			disableOutsidePointerEvents: !1,
			onCloseAutoFocus: (i) => {
				var s, l;
				(s = e.onCloseAutoFocus) == null || s.call(e, i), i.defaultPrevented || (r.current || (l = n.triggerRef.current) == null || l.focus(), i.preventDefault()), (r.current = !1), (o.current = !1);
			},
			onInteractOutside: (i) => {
				var a, u;
				(a = e.onInteractOutside) == null || a.call(e, i), i.defaultPrevented || ((r.current = !0), i.detail.originalEvent.type === 'pointerdown' && (o.current = !0));
				const s = i.target;
				((u = n.triggerRef.current) == null ? void 0 : u.contains(s)) && i.preventDefault(), i.detail.originalEvent.type === 'focusin' && o.current && i.preventDefault();
			},
		});
	}),
	qg = p.forwardRef((e, t) => {
		const {__scopeDialog: n, trapFocus: r, onOpenAutoFocus: o, onCloseAutoFocus: i, ...s} = e,
			l = wt(zn, n),
			a = p.useRef(null),
			u = we(t, a);
		return (
			Rh(),
			v.jsxs(v.Fragment, {
				children: [
					v.jsx(lc, {
						asChild: !0,
						loop: !0,
						trapped: r,
						onMountAutoFocus: o,
						onUnmountAutoFocus: i,
						children: v.jsx(js, {role: 'dialog', id: l.contentId, 'aria-describedby': l.descriptionId, 'aria-labelledby': l.titleId, 'data-state': Ec(l.open), ...s, ref: u, onDismiss: () => l.onOpenChange(!1)}),
					}),
					v.jsxs(v.Fragment, {children: [v.jsx(rb, {titleId: l.titleId}), v.jsx(ib, {contentRef: a, descriptionId: l.descriptionId})]}),
				],
			})
		);
	}),
	wc = 'DialogTitle',
	xc = p.forwardRef((e, t) => {
		const {__scopeDialog: n, ...r} = e,
			o = wt(wc, n);
		return v.jsx(pe.h2, {id: o.titleId, ...r, ref: t});
	});
xc.displayName = wc;
var Jg = 'DialogDescription',
	Sc = p.forwardRef((e, t) => {
		const {__scopeDialog: n, ...r} = e,
			o = wt(Jg, n);
		return v.jsx(pe.p, {id: o.descriptionId, ...r, ref: t});
	});
Sc.displayName = Jg;
var ev = 'DialogClose',
	tv = p.forwardRef((e, t) => {
		const {__scopeDialog: n, ...r} = e,
			o = wt(ev, n);
		return v.jsx(pe.button, {type: 'button', ...r, ref: t, onClick: B(e.onClick, () => o.onOpenChange(!1))});
	});
tv.displayName = ev;
function Ec(e) {
	return e ? 'open' : 'closed';
}
var nv = 'DialogTitleWarning',
	[Pk, rv] = Ww(nv, {contentName: zn, titleName: wc, docsSlug: 'dialog'}),
	rb = ({titleId: e}) => {
		const t = rv(nv),
			n = `\`${t.contentName}\` requires a \`${t.titleName}\` for the component to be accessible for screen reader users.

If you want to hide the \`${t.titleName}\`, you can wrap it with our VisuallyHidden component.

For more information, see https://radix-ui.com/primitives/docs/components/${t.docsSlug}`;
		return (
			p.useEffect(() => {
				e && (document.getElementById(e) || console.error(n));
			}, [n, e]),
			null
		);
	},
	ob = 'DialogDescriptionWarning',
	ib = ({contentRef: e, descriptionId: t}) => {
		const r = `Warning: Missing \`Description\` or \`aria-describedby={undefined}\` for {${rv(ob).contentName}}.`;
		return (
			p.useEffect(() => {
				var i;
				const o = (i = e.current) == null ? void 0 : i.getAttribute('aria-describedby');
				t && o && (document.getElementById(t) || console.warn(r));
			}, [r, e, t]),
			null
		);
	},
	ov = Vg,
	sb = Kg,
	iv = Qg,
	Ys = Xg,
	Qs = Zg,
	Xs = xc,
	Zs = Sc,
	sv = tv;
const lb = ov,
	ab = iv,
	lv = p.forwardRef(({className: e, ...t}, n) => v.jsx(Ys, {ref: n, className: he('fixed inset-0 z-50 bg-black/80  data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0', e), ...t}));
lv.displayName = Ys.displayName;
const av = p.forwardRef(({className: e, children: t, ...n}, r) =>
	v.jsxs(ab, {
		children: [
			v.jsx(lv, {}),
			v.jsxs(Qs, {
				ref: r,
				className: he(
					'fixed left-[50%] top-[50%] z-50 grid w-full max-w-lg translate-x-[-50%] translate-y-[-50%] gap-4 border bg-background p-6 shadow-lg duration-200 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=closed]:slide-out-to-left-1/2 data-[state=closed]:slide-out-to-top-[48%] data-[state=open]:slide-in-from-left-1/2 data-[state=open]:slide-in-from-top-[48%] sm:rounded-lg',
					e
				),
				...n,
				children: [
					t,
					v.jsxs(sv, {
						className:
							'absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-accent data-[state=open]:text-muted-foreground',
						children: [v.jsx(Bg, {className: 'h-4 w-4'}), v.jsx('span', {className: 'sr-only', children: 'Close'})],
					}),
				],
			}),
		],
	})
);
av.displayName = Qs.displayName;
const Cc = ({className: e, ...t}) => v.jsx('div', {className: he('flex flex-col space-y-1.5 text-center sm:text-left', e), ...t});
Cc.displayName = 'DialogHeader';
const uv = p.forwardRef(({className: e, ...t}, n) => v.jsx(Xs, {ref: n, className: he('text-lg font-semibold leading-none tracking-tight', e), ...t}));
uv.displayName = Xs.displayName;
const cv = p.forwardRef(({className: e, ...t}, n) => v.jsx(Zs, {ref: n, className: he('text-sm text-muted-foreground', e), ...t}));
cv.displayName = Zs.displayName;
const dv = 'NEW_SESSION',
	hf = {customer_id: '', title: 'New Conversation', agent_id: '', creation_utc: new Date().toLocaleString(), id: dv},
	ub = () => {
		var f;
		const {setAgentId: e, closeDialog: t, agents: n, setSessionId: r, setNewSession: o, customers: i} = Do(),
			[s, l] = p.useState('');
		p.useEffect(() => {
			n != null && n.length && n.length === 1 && a(n[0].id);
		}, []);
		const a = (d) => {
				var c;
				l(d), i.length < 2 && u((c = i == null ? void 0 : i[0]) == null ? void 0 : c.id, d);
			},
			u = (d, c) => {
				e(s || c || ''), o({...hf, agent_id: s, customer_id: d}), r(hf.id), t();
			};
		return v.jsxs('div', {
			className: 'h-full flex flex-col',
			children: [
				v.jsx(Cc, {
					children: v.jsx(uv, {
						children: v.jsxs('div', {
							className: 'h-[68px] w-full flex justify-between items-center ps-[30px] pe-[20px] border-b-[#EBECF0] border-b-[0.6px]',
							children: [
								v.jsx(cv, {className: 'text-[16px] font-bold', children: s ? 'Select a Customer' : 'Select an Agent'}),
								v.jsx('img', {role: 'button', tabIndex: 0, onKeyDown: Rr, onClick: t, className: 'cursor-pointer rounded-full hover:bg-[#F5F6F8] p-[10px]', src: 'icons/close.svg', alt: 'close', height: 30, width: 30}),
							],
						}),
					}),
				}),
				v.jsx('div', {
					className: 'flex flex-col overflow-auto relative flex-1',
					children:
						(f = s ? i : n) == null
							? void 0
							: f.map((d) =>
									v.jsxs(
										'div',
										{
											'data-testid': 'agent',
											tabIndex: 0,
											onKeyDown: Rr,
											role: 'button',
											onClick: () => (s ? u(d.id) : a(d.id)),
											className: Ku(
												'cursor-pointer hover:bg-[#FBFBFB] min-h-[78px] h-[78px] w-full border-b-[0.6px] border-b-solid border-b-[#EBECF0] flex items-center ps-[30px] pe-[20px]',
												d.name === '<guest>' && 'border-b-[#e1e2e3] border-b-[2px]'
											),
											children: [
												v.jsx(Ya, {agent: d, tooltip: !1}),
												v.jsxs('div', {
													children: [v.jsx('div', {className: 'text-[16px] font-medium', children: d.name === '<guest>' ? 'Guest' : d.name}), v.jsxs('div', {className: 'text-[14px] font-light text-[#A9A9A9]', children: ['(id=', d.id, ')']})],
												}),
											],
										},
										d.id
									)
							  ),
				}),
			],
		});
	},
	cb = ov,
	db = sb,
	fb = iv,
	fv = p.forwardRef(({className: e, ...t}, n) => v.jsx(Ys, {className: he('fixed inset-0 z-50 bg-black/80  data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0', e), ...t, ref: n}));
fv.displayName = Ys.displayName;
const pb = Ph('fixed z-50 gap-4 bg-background p-6 shadow-lg transition ease-in-out data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:duration-300 data-[state=open]:duration-500', {
		variants: {
			side: {
				top: 'inset-x-0 top-0 border-b data-[state=closed]:slide-out-to-top data-[state=open]:slide-in-from-top',
				bottom: 'inset-x-0 bottom-0 border-t data-[state=closed]:slide-out-to-bottom data-[state=open]:slide-in-from-bottom',
				left: 'inset-y-0 left-0 h-full w-3/4 border-r data-[state=closed]:slide-out-to-left data-[state=open]:slide-in-from-left sm:max-w-sm',
				right: 'inset-y-0 right-0 h-full w-3/4  border-l data-[state=closed]:slide-out-to-right data-[state=open]:slide-in-from-right sm:max-w-sm',
			},
		},
		defaultVariants: {side: 'right'},
	}),
	pv = p.forwardRef(({side: e = 'right', className: t, children: n, ...r}, o) =>
		v.jsxs(fb, {
			children: [
				v.jsx(fv, {}),
				v.jsxs(Qs, {
					ref: o,
					className: he(pb({side: e}), t),
					...r,
					children: [
						n,
						v.jsxs(sv, {
							className:
								'absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-secondary',
							children: [v.jsx(Bg, {className: 'h-4 w-4'}), v.jsx('span', {className: 'sr-only', children: 'Close'})],
						}),
					],
				}),
			],
		})
	);
pv.displayName = Qs.displayName;
const mv = ({className: e, ...t}) => v.jsx('div', {className: he('flex flex-col space-y-2 text-center sm:text-left', e), ...t});
mv.displayName = 'SheetHeader';
const hv = p.forwardRef(({className: e, ...t}, n) => v.jsx(Xs, {ref: n, className: he('text-lg font-semibold text-foreground', e), ...t}));
hv.displayName = Xs.displayName;
const gv = p.forwardRef(({className: e, ...t}, n) => v.jsx(Zs, {ref: n, className: he('text-sm text-muted-foreground', e), ...t}));
gv.displayName = Zs.displayName;
const $l = 'NEW_SESSION',
	mb = () => {
		const {setAgentId: e, openDialog: t} = Do(),
			[n, r] = p.useState(!1),
			{sessionId: o} = Do();
		p.useEffect(() => {
			n && r(!1);
		}, [o]);
		const i = () => {
			e(null), t('', v.jsx(ub, {}), {height: '536px', width: '604px'});
		};
		return v.jsx('div', {
			className: 'h-[70px] min-h-[70px] flex justify-between bg-white border-b-[0.6px] border-b-solid border-muted',
			children: v.jsxs('div', {
				className: 'w-[332px] max-mobile:w-full h-[70px] flex items-center justify-between',
				children: [
					v.jsx('div', {
						className: 'flex items-center min-[751px]:hidden',
						children: v.jsx('div', {
							children: v.jsxs(cb, {
								open: n,
								onOpenChange: () => r(!n),
								children: [
									v.jsx(db, {asChild: !0, onClick: () => r(!0), children: v.jsx(zC, {className: 'ms-[24px] cursor-pointer'})}),
									v.jsxs(pv, {side: 'left', className: 'w-fit px-0', children: [v.jsxs(mv, {children: [v.jsx(hv, {className: 'text-center'}), v.jsx(gv, {})]}), v.jsx(yv, {})]}),
								],
							}),
						}),
					}),
					v.jsxs('div', {
						className: 'flex items-center',
						children: [
							v.jsx('img', {src: '/chat/parlant-bubble-app-logo.svg', alt: 'logo', 'aria-hidden': !0, height: 17.9, width: 20.89, className: 'ms-[24px] me-[6px] max-mobile:ms-0'}),
							v.jsx('p', {className: 'text-[19.4px] font-bold', children: 'Parlant'}),
						],
					}),
					v.jsx('div', {
						className: 'group me-[24px]',
						children: v.jsx(cs, {
							value: 'New Session',
							side: 'right',
							children: v.jsxs('div', {
								children: [
									v.jsx('img', {onKeyDown: Rr, onClick: i, tabIndex: 1, role: 'button', src: 'icons/add.svg', alt: 'add session', height: 28, width: 28, className: 'cursor-pointer group-hover:hidden'}),
									v.jsx('img', {onKeyDown: Rr, onClick: i, tabIndex: 1, role: 'button', src: 'icons/add-filled.svg', alt: 'add session', height: 28, width: 28, className: 'cursor-pointer hidden group-hover:block'}),
								],
							}),
						}),
					}),
				],
			}),
		});
	},
	hb = ({session: e, closeDialog: t, deleteClicked: n}) =>
		v.jsxs('div', {
			'data-testid': 'deleteDialogContent',
			children: [
				v.jsx(vv, {session: e, disabled: !0}),
				v.jsxs('div', {
					className: 'h-[80px] flex items-center justify-end pe-[18px]',
					children: [
						v.jsx(Mo, {'data-testid': 'cancel-delete', onClick: t, className: 'hover:bg-[#EBE9F5] h-[46px] w-[96px] text-black bg-[#F2F0FC] rounded-[6px] py-[12px] px-[24px] me-[10px] text-[16px] font-normal', children: 'Cancel'}),
						v.jsx(Mo, {'data-testid': 'gradient-button', onClick: n, className: 'h-[46px] w-[161px] bg-[#213547] hover:bg-black rounded-[6px] py-[10px] px-[29.5px] text-[15px] font-medium', children: 'Delete Session'}),
					],
				}),
			],
		});
function vv({session: e, isSelected: t, refetch: n, editingTitle: r, setEditingTitle: o, tabIndex: i, disabled: s}) {
	const l = p.useRef(null),
		{setSessionId: a, setAgentId: u, setNewSession: f, agents: d, customers: c, setSessions: g, openDialog: w, closeDialog: y} = Do(),
		[S, m] = p.useState(new Map()),
		[h, x] = p.useState(new Map());
	p.useEffect(() => {
		t && (e.id === $l && !e.agent_id ? u(null) : u(e.agent_id));
	}, [t, u, e.id, e.agent_id, e.title]),
		p.useEffect(() => {
			d && m(new Map(d.map((R) => [R.id, R])));
		}, [d]),
		p.useEffect(() => {
			c && x(new Map(c.map((R) => [R.id, R])));
		}, [c]);
	const E = async (R) => {
			R.stopPropagation();
			const $ = (U) => {
				if ((y(), U.stopPropagation(), e.id === $l)) {
					f(null), a(null), u(null);
					return;
				}
				return V1(`sessions/${e.id}`)
					.then(() => {
						g((Q) => Q.filter((F) => F.id !== e.id)), t && (a(null), (document.title = 'Parlant')), Qr.success(`Session "${e.title}" deleted successfully`, {closeButton: !0});
					})
					.catch(() => {
						Qr.error('Something went wrong');
					});
			};
			w('Delete Session', v.jsx(hb, {closeDialog: y, deleteClicked: $, session: e}), {height: '230px', width: '480px'});
		},
		C = async (R) => {
			R.stopPropagation(),
				o == null || o(e.id),
				setTimeout(() => {
					var $;
					return ($ = l == null ? void 0 : l.current) == null ? void 0 : $.select();
				}, 0);
		},
		T = (R) => {
			var U, Q;
			R.stopPropagation();
			const $ = (Q = (U = l == null ? void 0 : l.current) == null ? void 0 : U.value) == null ? void 0 : Q.trim();
			if ($) {
				if (e.id === $l) {
					o == null || o(null), f((F) => F && {...F, title: $}), Qr.success('title changed successfully', {closeButton: !0});
					return;
				}
				W1(`sessions/${e.id}`, {title: $})
					.then(() => {
						o == null || o(null), n == null || n(), Qr.success('title changed successfully', {closeButton: !0});
					})
					.catch(() => {
						Qr.error('Something went wrong');
					});
			}
		},
		k = (R) => {
			R.stopPropagation(), o == null || o(null);
		},
		N = (R) => {
			R.key === 'Enter' && T(R);
		},
		D = [
			{title: 'rename', onClick: C, imgPath: 'icons/rename.svg'},
			{title: 'delete', onClick: E, imgPath: 'icons/delete.svg'},
		],
		M = S.get(e.agent_id),
		z = h.get(e.customer_id);
	return v.jsxs(
		'div',
		{
			'data-testid': 'session',
			role: 'button',
			tabIndex: i,
			onKeyDown: Rr,
			onClick: () => !s && !r && a(e.id),
			className:
				'bg-white animate-fade-in text-[14px] font-ubuntu-sans justify-between font-medium border-b-[0.6px] border-b-solid border-muted cursor-pointer p-1 flex items-center ps-[8px] min-h-[80px] h-[80px] ml-0 mr-0 ' +
				(r === e.id ? ZC.editSession + ' !p-[4px_2px] ' : r ? ' opacity-[33%] ' : ' hover:bg-main ') +
				(t && r !== e.id ? '!bg-[#FAF9FF]' : '') +
				(s ? ' pointer-events-none' : ''),
			children: [
				v.jsxs('div', {
					className: 'flex-1 whitespace-nowrap overflow-hidden max-w-[202px] ms-[16px] h-[39px]',
					children: [
						r !== e.id &&
							v.jsxs('div', {
								className: 'overflow-hidden overflow-ellipsis flex items-center',
								children: [
									v.jsx('div', {children: v.jsx(Ya, {agent: M || {id: '', name: 'N/A'}, customer: z || {id: '', name: 'N/A'}})}),
									v.jsxs('div', {
										className: zm(!M && 'opacity-50'),
										children: [
											e.title,
											v.jsxs('small', {className: 'text-[12px] text-[#A9A9A9] font-light flex gap-[6px]', children: [YC(e.creation_utc), v.jsx('img', {src: 'icons/dot-saparetor.svg', alt: '', height: 18, width: 3}), QC(e.creation_utc)]}),
										],
									}),
								],
							}),
						r === e.id &&
							v.jsxs('div', {
								className: 'flex items-center ps-[6px]',
								children: [
									v.jsx('div', {children: M && v.jsx(Ya, {agent: M})}),
									v.jsx(Hm, {'data-testid': 'sessionTitle', ref: l, onKeyUp: N, onClick: (R) => R.stopPropagation(), autoFocus: !0, defaultValue: e.title, className: 'box-shadow-none border-none bg-[#F5F6F8] text-foreground h-fit p-1 ms-[6px]'}),
								],
							}),
					],
				}),
				v.jsxs('div', {
					className: 'h-[39px] flex items-center',
					children: [
						!s &&
							r !== e.id &&
							v.jsxs($C, {
								children: [
									v.jsx(BC, {
										disabled: !!r,
										'data-testid': 'menu-button',
										tabIndex: -1,
										onClick: (R) => R.stopPropagation(),
										children: v.jsx('div', {tabIndex: i, role: 'button', className: 'rounded-full me-[24px]', onClick: (R) => R.stopPropagation(), children: v.jsx('img', {src: 'icons/more.svg', alt: 'more', height: 14, width: 14})}),
									}),
									v.jsx(Ug, {
										align: 'start',
										children: D.map((R) =>
											v.jsxs(
												Hg,
												{
													tabIndex: 0,
													onClick: R.onClick,
													className: 'gap-0 font-medium text-[14px] font-ubuntu-sans capitalize hover:!bg-[#FAF9FF]',
													children: [v.jsx('img', {'data-testid': R.title, src: R.imgPath, height: 16, width: 18, className: 'me-[8px]', alt: ''}), R.title],
												},
												R.title
											)
										),
									}),
								],
							}),
						r == e.id &&
							v.jsxs('div', {
								className: 'me-[18px]',
								children: [
									v.jsx(cs, {value: 'Cancel', children: v.jsx(Mo, {'data-testid': 'cancel', variant: 'ghost', className: 'w-[28px] h-[28px] p-[8px] rounded-full', onClick: k, children: v.jsx('img', {src: 'icons/cancel.svg', alt: 'cancel'})})}),
									v.jsx(cs, {value: 'Save', children: v.jsx(Mo, {variant: 'ghost', className: 'w-[28px] h-[28px] p-[8px] rounded-full', onClick: T, children: v.jsx('img', {src: 'icons/save.svg', alt: 'cancel'})})}),
								],
							}),
					],
				}),
			],
		},
		e.id
	);
}
const gb = ({children: e, height: t, className: n}) => {
	const r = p.useRef(null),
		[o, i] = p.useState([]),
		s = p.useRef(null);
	return (
		p.useEffect(() => {
			var a;
			s.current = new IntersectionObserver((u) => {
				u.forEach((f) => {
					const d = parseInt(f.target.getAttribute('data-index') || '', 10);
					isNaN(d) || (f.isIntersecting ? i((c) => [...new Set([...c, d])]) : i((c) => c.filter((g) => g !== d)));
				});
			});
			const l = (a = r.current) == null ? void 0 : a.children;
			return (
				l &&
					Array.from(l).forEach((u, f) => {
						var d;
						u.setAttribute('data-index', f.toString()), (d = s.current) == null || d.observe(u);
					}),
				() => {
					var u;
					return (u = s.current) == null ? void 0 : u.disconnect();
				}
			);
		}, [e]),
		v.jsx('div', {className: 'scroll-container ' + n, ref: r, children: e.map((l, a) => v.jsx('div', {'data-index': a, className: 'item' + (o.includes(a) ? '' : ` h-[${t ?? '1px'}]`), children: o.includes(a) ? l : ''}, a))})
	);
};
function yv() {
	const [e, t] = p.useState(null),
		{sessionId: n, setSessions: r, sessions: o, setAgents: i, setCustomers: s} = Do(),
		{data: l, ErrorTemplate: a, loading: u, refetch: f} = Nl('sessions'),
		{data: d} = Nl('agents'),
		{data: c} = Nl('customers');
	return (
		p.useEffect(() => {
			d && i(d);
		}, [d]),
		p.useEffect(() => {
			c && s(c);
		}, [c]),
		p.useEffect(() => {
			l && r(l);
		}, [l]),
		v.jsx('div', {
			className: 'flex flex-col items-center h-full',
			children: v.jsxs('div', {
				'data-testid': 'sessions',
				className: 'bg-white flex-1 justify-center w-[332px] overflow-auto',
				children: [
					u && !(o != null && o.length) && v.jsx('div', {children: 'loading...'}),
					v.jsx(gb, {
						height: '80px',
						className: 'flex flex-col-reverse',
						children: o.map((g, w) => v.jsx(vv, {'data-testid': 'session', tabIndex: o.length - w, editingTitle: e, setEditingTitle: t, isSelected: g.id === n, refetch: f, session: g}, g.id)),
					}),
					a && v.jsx(a, {}),
				],
			}),
		})
	);
}
class vb extends p.Component {
	constructor(t) {
		super(t), (this.state = {hasError: !1});
	}
	static getDerivedStateFromError() {
		return {hasError: !0};
	}
	componentDidCatch(t) {
		this.setState({errorStack: t.stack});
	}
	render() {
		return this.state.hasError
			? v.jsxs('div', {
					className: 'flex bg-main items-center justify-center h-screen flex-col',
					children: [
						v.jsx('img', {src: '/chat/parlant-bubble-app-logo.svg', alt: 'Logo', height: 200, width: 200, className: 'mb-[10px]'}),
						v.jsx('h1', {className: 'text-[20px]', children: 'Oops! Something went wrong'}),
						v.jsxs('p', {className: 'text-center', children: ['We apologize for the inconvenience. Please try again later, or ', v.jsx('a', {href: '/', className: 'underline', children: 'try again now'}), '.']}),
						v.jsx('div', {
							className: 'flex justify-center max-h-[300px] mt-[40px] bg-[#f0eeee] rounded-[10px] p-[10px]  break-words border border-solid border-[#dedcdc]',
							children: v.jsx('code', {className: 'max-h-[300px] w-[600px] max-w-[80vw] overflow-auto', children: this.state.errorStack}),
						}),
					],
			  })
			: this.props.children;
	}
}
const yb = () => {
	const [e, t] = p.useState(null),
		[n, r] = p.useState(null),
		[o, i] = p.useState({height: '', width: ''}),
		[s, l] = p.useState(null),
		a = (d, c, g, w = null) => {
			t(d), r(c), i({height: g.height, width: g.width}), w && l(w);
		},
		u = (d) => {
			d == null || d.stopPropagation(), r(null), t(null), s == null || s(), l(null);
		};
	return {
		openDialog: a,
		DialogComponent: () =>
			v.jsx(lb, {
				open: !!n,
				children: v.jsx(av, {
					'data-testid': 'dialog',
					style: {maxHeight: o.height, width: o.width},
					className: '[&>button]:hidden p-0 h-[80%] font-ubuntu-sans bg-white block max-w-[95%]',
					children: v.jsxs('div', {
						className: 'bg-white h-full rounded-[12px] flex flex-col',
						children: [
							v.jsx(Cc, {
								className: Ku(!e && 'hidden'),
								children: v.jsx(xc, {
									children: v.jsxs('div', {
										className: 'h-[68px] w-full flex justify-between items-center ps-[30px] pe-[20px] border-b-[#EBECF0] border-b-[0.6px]',
										children: [
											v.jsx(Sc, {className: 'text-[16px] font-bold', children: e}),
											v.jsx('img', {role: 'button', tabIndex: 0, onKeyDown: Rr, onClick: u, className: 'cursor-pointer rounded-full hover:bg-[#F5F6F8] p-[10px]', src: 'icons/close.svg', alt: 'close', height: 30, width: 30}),
										],
									}),
								}),
							}),
							v.jsx('div', {className: 'overflow-auto flex-1', children: n}),
						],
					}),
				}),
			}),
		closeDialog: u,
	};
};
var wv = {exports: {}},
	wb = 'SECRET_DO_NOT_PASS_THIS_OR_YOU_WILL_BE_FIRED',
	xb = wb,
	Sb = xb;
function xv() {}
function Sv() {}
Sv.resetWarningCache = xv;
var Eb = function () {
	function e(r, o, i, s, l, a) {
		if (a !== Sb) {
			var u = new Error('Calling PropTypes validators directly is not supported by the `prop-types` package. Use PropTypes.checkPropTypes() to call them. Read more at http://fb.me/use-check-prop-types');
			throw ((u.name = 'Invariant Violation'), u);
		}
	}
	e.isRequired = e;
	function t() {
		return e;
	}
	var n = {
		array: e,
		bigint: e,
		bool: e,
		func: e,
		number: e,
		object: e,
		string: e,
		symbol: e,
		any: e,
		arrayOf: t,
		element: e,
		elementType: e,
		instanceOf: t,
		node: e,
		objectOf: t,
		oneOf: t,
		oneOfType: t,
		shape: t,
		exact: t,
		checkPropTypes: Sv,
		resetWarningCache: xv,
	};
	return (n.PropTypes = n), n;
};
wv.exports = Eb();
var Cb = wv.exports;
const se = Ar(Cb);
function bb(e) {
	return e && typeof e == 'object' && 'default' in e ? e.default : e;
}
var Ev = p,
	kb = bb(Ev);
function gf(e, t, n) {
	return t in e ? Object.defineProperty(e, t, {value: n, enumerable: !0, configurable: !0, writable: !0}) : (e[t] = n), e;
}
function Tb(e, t) {
	(e.prototype = Object.create(t.prototype)), (e.prototype.constructor = e), (e.__proto__ = t);
}
var Pb = !!(typeof window < 'u' && window.document && window.document.createElement);
function Nb(e, t, n) {
	if (typeof e != 'function') throw new Error('Expected reducePropsToState to be a function.');
	if (typeof t != 'function') throw new Error('Expected handleStateChangeOnClient to be a function.');
	if (typeof n < 'u' && typeof n != 'function') throw new Error('Expected mapStateOnServer to either be undefined or a function.');
	function r(o) {
		return o.displayName || o.name || 'Component';
	}
	return function (i) {
		if (typeof i != 'function') throw new Error('Expected WrappedComponent to be a React component.');
		var s = [],
			l;
		function a() {
			(l = e(
				s.map(function (f) {
					return f.props;
				})
			)),
				u.canUseDOM ? t(l) : n && (l = n(l));
		}
		var u = (function (f) {
			Tb(d, f);
			function d() {
				return f.apply(this, arguments) || this;
			}
			(d.peek = function () {
				return l;
			}),
				(d.rewind = function () {
					if (d.canUseDOM) throw new Error('You may only call rewind() on the server. Call peek() to read the current state.');
					var w = l;
					return (l = void 0), (s = []), w;
				});
			var c = d.prototype;
			return (
				(c.UNSAFE_componentWillMount = function () {
					s.push(this), a();
				}),
				(c.componentDidUpdate = function () {
					a();
				}),
				(c.componentWillUnmount = function () {
					var w = s.indexOf(this);
					s.splice(w, 1), a();
				}),
				(c.render = function () {
					return kb.createElement(i, this.props);
				}),
				d
			);
		})(Ev.PureComponent);
		return gf(u, 'displayName', 'SideEffect(' + r(i) + ')'), gf(u, 'canUseDOM', Pb), u;
	};
}
var _b = Nb;
const Rb = Ar(_b);
var Ab = typeof Element < 'u',
	Mb = typeof Map == 'function',
	Ob = typeof Set == 'function',
	jb = typeof ArrayBuffer == 'function' && !!ArrayBuffer.isView;
function Ii(e, t) {
	if (e === t) return !0;
	if (e && t && typeof e == 'object' && typeof t == 'object') {
		if (e.constructor !== t.constructor) return !1;
		var n, r, o;
		if (Array.isArray(e)) {
			if (((n = e.length), n != t.length)) return !1;
			for (r = n; r-- !== 0; ) if (!Ii(e[r], t[r])) return !1;
			return !0;
		}
		var i;
		if (Mb && e instanceof Map && t instanceof Map) {
			if (e.size !== t.size) return !1;
			for (i = e.entries(); !(r = i.next()).done; ) if (!t.has(r.value[0])) return !1;
			for (i = e.entries(); !(r = i.next()).done; ) if (!Ii(r.value[1], t.get(r.value[0]))) return !1;
			return !0;
		}
		if (Ob && e instanceof Set && t instanceof Set) {
			if (e.size !== t.size) return !1;
			for (i = e.entries(); !(r = i.next()).done; ) if (!t.has(r.value[0])) return !1;
			return !0;
		}
		if (jb && ArrayBuffer.isView(e) && ArrayBuffer.isView(t)) {
			if (((n = e.length), n != t.length)) return !1;
			for (r = n; r-- !== 0; ) if (e[r] !== t[r]) return !1;
			return !0;
		}
		if (e.constructor === RegExp) return e.source === t.source && e.flags === t.flags;
		if (e.valueOf !== Object.prototype.valueOf && typeof e.valueOf == 'function' && typeof t.valueOf == 'function') return e.valueOf() === t.valueOf();
		if (e.toString !== Object.prototype.toString && typeof e.toString == 'function' && typeof t.toString == 'function') return e.toString() === t.toString();
		if (((o = Object.keys(e)), (n = o.length), n !== Object.keys(t).length)) return !1;
		for (r = n; r-- !== 0; ) if (!Object.prototype.hasOwnProperty.call(t, o[r])) return !1;
		if (Ab && e instanceof Element) return !1;
		for (r = n; r-- !== 0; ) if (!((o[r] === '_owner' || o[r] === '__v' || o[r] === '__o') && e.$$typeof) && !Ii(e[o[r]], t[o[r]])) return !1;
		return !0;
	}
	return e !== e && t !== t;
}
var Ib = function (t, n) {
	try {
		return Ii(t, n);
	} catch (r) {
		if ((r.message || '').match(/stack|recursion/i)) return console.warn('react-fast-compare cannot handle circular refs'), !1;
		throw r;
	}
};
const Db = Ar(Ib);
/*
object-assign
(c) Sindre Sorhus
@license MIT
*/ var vf = Object.getOwnPropertySymbols,
	Lb = Object.prototype.hasOwnProperty,
	Fb = Object.prototype.propertyIsEnumerable;
function zb(e) {
	if (e == null) throw new TypeError('Object.assign cannot be called with null or undefined');
	return Object(e);
}
function $b() {
	try {
		if (!Object.assign) return !1;
		var e = new String('abc');
		if (((e[5] = 'de'), Object.getOwnPropertyNames(e)[0] === '5')) return !1;
		for (var t = {}, n = 0; n < 10; n++) t['_' + String.fromCharCode(n)] = n;
		var r = Object.getOwnPropertyNames(t).map(function (i) {
			return t[i];
		});
		if (r.join('') !== '0123456789') return !1;
		var o = {};
		return (
			'abcdefghijklmnopqrst'.split('').forEach(function (i) {
				o[i] = i;
			}),
			Object.keys(Object.assign({}, o)).join('') === 'abcdefghijklmnopqrst'
		);
	} catch {
		return !1;
	}
}
var Bb = $b()
	? Object.assign
	: function (e, t) {
			for (var n, r = zb(e), o, i = 1; i < arguments.length; i++) {
				n = Object(arguments[i]);
				for (var s in n) Lb.call(n, s) && (r[s] = n[s]);
				if (vf) {
					o = vf(n);
					for (var l = 0; l < o.length; l++) Fb.call(n, o[l]) && (r[o[l]] = n[o[l]]);
				}
			}
			return r;
	  };
const Ub = Ar(Bb);
var Rn = {BODY: 'bodyAttributes', HTML: 'htmlAttributes', TITLE: 'titleAttributes'},
	H = {BASE: 'base', BODY: 'body', HEAD: 'head', HTML: 'html', LINK: 'link', META: 'meta', NOSCRIPT: 'noscript', SCRIPT: 'script', STYLE: 'style', TITLE: 'title'};
Object.keys(H).map(function (e) {
	return H[e];
});
var le = {CHARSET: 'charset', CSS_TEXT: 'cssText', HREF: 'href', HTTPEQUIV: 'http-equiv', INNER_HTML: 'innerHTML', ITEM_PROP: 'itemprop', NAME: 'name', PROPERTY: 'property', REL: 'rel', SRC: 'src', TARGET: 'target'},
	ms = {accesskey: 'accessKey', charset: 'charSet', class: 'className', contenteditable: 'contentEditable', contextmenu: 'contextMenu', 'http-equiv': 'httpEquiv', itemprop: 'itemProp', tabindex: 'tabIndex'},
	Io = {DEFAULT_TITLE: 'defaultTitle', DEFER: 'defer', ENCODE_SPECIAL_CHARACTERS: 'encodeSpecialCharacters', ON_CHANGE_CLIENT_STATE: 'onChangeClientState', TITLE_TEMPLATE: 'titleTemplate'},
	Hb = Object.keys(ms).reduce(function (e, t) {
		return (e[ms[t]] = t), e;
	}, {}),
	Wb = [H.NOSCRIPT, H.SCRIPT, H.STYLE],
	ft = 'data-react-helmet',
	Vb =
		typeof Symbol == 'function' && typeof Symbol.iterator == 'symbol'
			? function (e) {
					return typeof e;
			  }
			: function (e) {
					return e && typeof Symbol == 'function' && e.constructor === Symbol && e !== Symbol.prototype ? 'symbol' : typeof e;
			  },
	Gb = function (e, t) {
		if (!(e instanceof t)) throw new TypeError('Cannot call a class as a function');
	},
	Kb = (function () {
		function e(t, n) {
			for (var r = 0; r < n.length; r++) {
				var o = n[r];
				(o.enumerable = o.enumerable || !1), (o.configurable = !0), 'value' in o && (o.writable = !0), Object.defineProperty(t, o.key, o);
			}
		}
		return function (t, n, r) {
			return n && e(t.prototype, n), r && e(t, r), t;
		};
	})(),
	Fe =
		Object.assign ||
		function (e) {
			for (var t = 1; t < arguments.length; t++) {
				var n = arguments[t];
				for (var r in n) Object.prototype.hasOwnProperty.call(n, r) && (e[r] = n[r]);
			}
			return e;
		},
	Yb = function (e, t) {
		if (typeof t != 'function' && t !== null) throw new TypeError('Super expression must either be null or a function, not ' + typeof t);
		(e.prototype = Object.create(t && t.prototype, {constructor: {value: e, enumerable: !1, writable: !0, configurable: !0}})), t && (Object.setPrototypeOf ? Object.setPrototypeOf(e, t) : (e.__proto__ = t));
	},
	yf = function (e, t) {
		var n = {};
		for (var r in e) t.indexOf(r) >= 0 || (Object.prototype.hasOwnProperty.call(e, r) && (n[r] = e[r]));
		return n;
	},
	Qb = function (e, t) {
		if (!e) throw new ReferenceError("this hasn't been initialised - super() hasn't been called");
		return t && (typeof t == 'object' || typeof t == 'function') ? t : e;
	},
	Qa = function (t) {
		var n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : !0;
		return n === !1 ? String(t) : String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#x27;');
	},
	Xb = function (t) {
		var n = xr(t, H.TITLE),
			r = xr(t, Io.TITLE_TEMPLATE);
		if (r && n)
			return r.replace(/%s/g, function () {
				return Array.isArray(n) ? n.join('') : n;
			});
		var o = xr(t, Io.DEFAULT_TITLE);
		return n || o || void 0;
	},
	Zb = function (t) {
		return xr(t, Io.ON_CHANGE_CLIENT_STATE) || function () {};
	},
	Bl = function (t, n) {
		return n
			.filter(function (r) {
				return typeof r[t] < 'u';
			})
			.map(function (r) {
				return r[t];
			})
			.reduce(function (r, o) {
				return Fe({}, r, o);
			}, {});
	},
	qb = function (t, n) {
		return n
			.filter(function (r) {
				return typeof r[H.BASE] < 'u';
			})
			.map(function (r) {
				return r[H.BASE];
			})
			.reverse()
			.reduce(function (r, o) {
				if (!r.length)
					for (var i = Object.keys(o), s = 0; s < i.length; s++) {
						var l = i[s],
							a = l.toLowerCase();
						if (t.indexOf(a) !== -1 && o[a]) return r.concat(o);
					}
				return r;
			}, []);
	},
	Xr = function (t, n, r) {
		var o = {};
		return r
			.filter(function (i) {
				return Array.isArray(i[t]) ? !0 : (typeof i[t] < 'u' && nk('Helmet: ' + t + ' should be of type "Array". Instead found type "' + Vb(i[t]) + '"'), !1);
			})
			.map(function (i) {
				return i[t];
			})
			.reverse()
			.reduce(function (i, s) {
				var l = {};
				s.filter(function (c) {
					for (var g = void 0, w = Object.keys(c), y = 0; y < w.length; y++) {
						var S = w[y],
							m = S.toLowerCase();
						n.indexOf(m) !== -1 && !(g === le.REL && c[g].toLowerCase() === 'canonical') && !(m === le.REL && c[m].toLowerCase() === 'stylesheet') && (g = m),
							n.indexOf(S) !== -1 && (S === le.INNER_HTML || S === le.CSS_TEXT || S === le.ITEM_PROP) && (g = S);
					}
					if (!g || !c[g]) return !1;
					var h = c[g].toLowerCase();
					return o[g] || (o[g] = {}), l[g] || (l[g] = {}), o[g][h] ? !1 : ((l[g][h] = !0), !0);
				})
					.reverse()
					.forEach(function (c) {
						return i.push(c);
					});
				for (var a = Object.keys(l), u = 0; u < a.length; u++) {
					var f = a[u],
						d = Ub({}, o[f], l[f]);
					o[f] = d;
				}
				return i;
			}, [])
			.reverse();
	},
	xr = function (t, n) {
		for (var r = t.length - 1; r >= 0; r--) {
			var o = t[r];
			if (o.hasOwnProperty(n)) return o[n];
		}
		return null;
	},
	Jb = function (t) {
		return {
			baseTag: qb([le.HREF, le.TARGET], t),
			bodyAttributes: Bl(Rn.BODY, t),
			defer: xr(t, Io.DEFER),
			encode: xr(t, Io.ENCODE_SPECIAL_CHARACTERS),
			htmlAttributes: Bl(Rn.HTML, t),
			linkTags: Xr(H.LINK, [le.REL, le.HREF], t),
			metaTags: Xr(H.META, [le.NAME, le.CHARSET, le.HTTPEQUIV, le.PROPERTY, le.ITEM_PROP], t),
			noscriptTags: Xr(H.NOSCRIPT, [le.INNER_HTML], t),
			onChangeClientState: Zb(t),
			scriptTags: Xr(H.SCRIPT, [le.SRC, le.INNER_HTML], t),
			styleTags: Xr(H.STYLE, [le.CSS_TEXT], t),
			title: Xb(t),
			titleAttributes: Bl(Rn.TITLE, t),
		};
	},
	Xa = (function () {
		var e = Date.now();
		return function (t) {
			var n = Date.now();
			n - e > 16
				? ((e = n), t(n))
				: setTimeout(function () {
						Xa(t);
				  }, 0);
		};
	})(),
	wf = function (t) {
		return clearTimeout(t);
	},
	ek = typeof window < 'u' ? (window.requestAnimationFrame && window.requestAnimationFrame.bind(window)) || window.webkitRequestAnimationFrame || window.mozRequestAnimationFrame || Xa : global.requestAnimationFrame || Xa,
	tk = typeof window < 'u' ? window.cancelAnimationFrame || window.webkitCancelAnimationFrame || window.mozCancelAnimationFrame || wf : global.cancelAnimationFrame || wf,
	nk = function (t) {
		return console && typeof console.warn == 'function' && console.warn(t);
	},
	Zr = null,
	rk = function (t) {
		Zr && tk(Zr),
			t.defer
				? (Zr = ek(function () {
						xf(t, function () {
							Zr = null;
						});
				  }))
				: (xf(t), (Zr = null));
	},
	xf = function (t, n) {
		var r = t.baseTag,
			o = t.bodyAttributes,
			i = t.htmlAttributes,
			s = t.linkTags,
			l = t.metaTags,
			a = t.noscriptTags,
			u = t.onChangeClientState,
			f = t.scriptTags,
			d = t.styleTags,
			c = t.title,
			g = t.titleAttributes;
		Za(H.BODY, o), Za(H.HTML, i), ok(c, g);
		var w = {baseTag: qn(H.BASE, r), linkTags: qn(H.LINK, s), metaTags: qn(H.META, l), noscriptTags: qn(H.NOSCRIPT, a), scriptTags: qn(H.SCRIPT, f), styleTags: qn(H.STYLE, d)},
			y = {},
			S = {};
		Object.keys(w).forEach(function (m) {
			var h = w[m],
				x = h.newTags,
				E = h.oldTags;
			x.length && (y[m] = x), E.length && (S[m] = w[m].oldTags);
		}),
			n && n(),
			u(t, y, S);
	},
	Cv = function (t) {
		return Array.isArray(t) ? t.join('') : t;
	},
	ok = function (t, n) {
		typeof t < 'u' && document.title !== t && (document.title = Cv(t)), Za(H.TITLE, n);
	},
	Za = function (t, n) {
		var r = document.getElementsByTagName(t)[0];
		if (r) {
			for (var o = r.getAttribute(ft), i = o ? o.split(',') : [], s = [].concat(i), l = Object.keys(n), a = 0; a < l.length; a++) {
				var u = l[a],
					f = n[u] || '';
				r.getAttribute(u) !== f && r.setAttribute(u, f), i.indexOf(u) === -1 && i.push(u);
				var d = s.indexOf(u);
				d !== -1 && s.splice(d, 1);
			}
			for (var c = s.length - 1; c >= 0; c--) r.removeAttribute(s[c]);
			i.length === s.length ? r.removeAttribute(ft) : r.getAttribute(ft) !== l.join(',') && r.setAttribute(ft, l.join(','));
		}
	},
	qn = function (t, n) {
		var r = document.head || document.querySelector(H.HEAD),
			o = r.querySelectorAll(t + '[' + ft + ']'),
			i = Array.prototype.slice.call(o),
			s = [],
			l = void 0;
		return (
			n &&
				n.length &&
				n.forEach(function (a) {
					var u = document.createElement(t);
					for (var f in a)
						if (a.hasOwnProperty(f))
							if (f === le.INNER_HTML) u.innerHTML = a.innerHTML;
							else if (f === le.CSS_TEXT) u.styleSheet ? (u.styleSheet.cssText = a.cssText) : u.appendChild(document.createTextNode(a.cssText));
							else {
								var d = typeof a[f] > 'u' ? '' : a[f];
								u.setAttribute(f, d);
							}
					u.setAttribute(ft, 'true'),
						i.some(function (c, g) {
							return (l = g), u.isEqualNode(c);
						})
							? i.splice(l, 1)
							: s.push(u);
				}),
			i.forEach(function (a) {
				return a.parentNode.removeChild(a);
			}),
			s.forEach(function (a) {
				return r.appendChild(a);
			}),
			{oldTags: i, newTags: s}
		);
	},
	bv = function (t) {
		return Object.keys(t).reduce(function (n, r) {
			var o = typeof t[r] < 'u' ? r + '="' + t[r] + '"' : '' + r;
			return n ? n + ' ' + o : o;
		}, '');
	},
	ik = function (t, n, r, o) {
		var i = bv(r),
			s = Cv(n);
		return i ? '<' + t + ' ' + ft + '="true" ' + i + '>' + Qa(s, o) + '</' + t + '>' : '<' + t + ' ' + ft + '="true">' + Qa(s, o) + '</' + t + '>';
	},
	sk = function (t, n, r) {
		return n.reduce(function (o, i) {
			var s = Object.keys(i)
					.filter(function (u) {
						return !(u === le.INNER_HTML || u === le.CSS_TEXT);
					})
					.reduce(function (u, f) {
						var d = typeof i[f] > 'u' ? f : f + '="' + Qa(i[f], r) + '"';
						return u ? u + ' ' + d : d;
					}, ''),
				l = i.innerHTML || i.cssText || '',
				a = Wb.indexOf(t) === -1;
			return o + '<' + t + ' ' + ft + '="true" ' + s + (a ? '/>' : '>' + l + '</' + t + '>');
		}, '');
	},
	kv = function (t) {
		var n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {};
		return Object.keys(t).reduce(function (r, o) {
			return (r[ms[o] || o] = t[o]), r;
		}, n);
	},
	lk = function (t) {
		var n = arguments.length > 1 && arguments[1] !== void 0 ? arguments[1] : {};
		return Object.keys(t).reduce(function (r, o) {
			return (r[Hb[o] || o] = t[o]), r;
		}, n);
	},
	ak = function (t, n, r) {
		var o,
			i = ((o = {key: n}), (o[ft] = !0), o),
			s = kv(r, i);
		return [A.createElement(H.TITLE, s, n)];
	},
	uk = function (t, n) {
		return n.map(function (r, o) {
			var i,
				s = ((i = {key: o}), (i[ft] = !0), i);
			return (
				Object.keys(r).forEach(function (l) {
					var a = ms[l] || l;
					if (a === le.INNER_HTML || a === le.CSS_TEXT) {
						var u = r.innerHTML || r.cssText;
						s.dangerouslySetInnerHTML = {__html: u};
					} else s[a] = r[l];
				}),
				A.createElement(t, s)
			);
		});
	},
	At = function (t, n, r) {
		switch (t) {
			case H.TITLE:
				return {
					toComponent: function () {
						return ak(t, n.title, n.titleAttributes);
					},
					toString: function () {
						return ik(t, n.title, n.titleAttributes, r);
					},
				};
			case Rn.BODY:
			case Rn.HTML:
				return {
					toComponent: function () {
						return kv(n);
					},
					toString: function () {
						return bv(n);
					},
				};
			default:
				return {
					toComponent: function () {
						return uk(t, n);
					},
					toString: function () {
						return sk(t, n, r);
					},
				};
		}
	},
	Tv = function (t) {
		var n = t.baseTag,
			r = t.bodyAttributes,
			o = t.encode,
			i = t.htmlAttributes,
			s = t.linkTags,
			l = t.metaTags,
			a = t.noscriptTags,
			u = t.scriptTags,
			f = t.styleTags,
			d = t.title,
			c = d === void 0 ? '' : d,
			g = t.titleAttributes;
		return {
			base: At(H.BASE, n, o),
			bodyAttributes: At(Rn.BODY, r, o),
			htmlAttributes: At(Rn.HTML, i, o),
			link: At(H.LINK, s, o),
			meta: At(H.META, l, o),
			noscript: At(H.NOSCRIPT, a, o),
			script: At(H.SCRIPT, u, o),
			style: At(H.STYLE, f, o),
			title: At(H.TITLE, {title: c, titleAttributes: g}, o),
		};
	},
	ck = function (t) {
		var n, r;
		return (
			(r = n =
				(function (o) {
					Yb(i, o);
					function i() {
						return Gb(this, i), Qb(this, o.apply(this, arguments));
					}
					return (
						(i.prototype.shouldComponentUpdate = function (l) {
							return !Db(this.props, l);
						}),
						(i.prototype.mapNestedChildrenToProps = function (l, a) {
							if (!a) return null;
							switch (l.type) {
								case H.SCRIPT:
								case H.NOSCRIPT:
									return {innerHTML: a};
								case H.STYLE:
									return {cssText: a};
							}
							throw new Error('<' + l.type + ' /> elements are self-closing and can not contain children. Refer to our API for more information.');
						}),
						(i.prototype.flattenArrayTypeChildren = function (l) {
							var a,
								u = l.child,
								f = l.arrayTypeChildren,
								d = l.newChildProps,
								c = l.nestedChildren;
							return Fe({}, f, ((a = {}), (a[u.type] = [].concat(f[u.type] || [], [Fe({}, d, this.mapNestedChildrenToProps(u, c))])), a));
						}),
						(i.prototype.mapObjectTypeChildren = function (l) {
							var a,
								u,
								f = l.child,
								d = l.newProps,
								c = l.newChildProps,
								g = l.nestedChildren;
							switch (f.type) {
								case H.TITLE:
									return Fe({}, d, ((a = {}), (a[f.type] = g), (a.titleAttributes = Fe({}, c)), a));
								case H.BODY:
									return Fe({}, d, {bodyAttributes: Fe({}, c)});
								case H.HTML:
									return Fe({}, d, {htmlAttributes: Fe({}, c)});
							}
							return Fe({}, d, ((u = {}), (u[f.type] = Fe({}, c)), u));
						}),
						(i.prototype.mapArrayTypeChildrenToProps = function (l, a) {
							var u = Fe({}, a);
							return (
								Object.keys(l).forEach(function (f) {
									var d;
									u = Fe({}, u, ((d = {}), (d[f] = l[f]), d));
								}),
								u
							);
						}),
						(i.prototype.warnOnInvalidChildren = function (l, a) {
							return !0;
						}),
						(i.prototype.mapChildrenToProps = function (l, a) {
							var u = this,
								f = {};
							return (
								A.Children.forEach(l, function (d) {
									if (!(!d || !d.props)) {
										var c = d.props,
											g = c.children,
											w = yf(c, ['children']),
											y = lk(w);
										switch ((u.warnOnInvalidChildren(d, g), d.type)) {
											case H.LINK:
											case H.META:
											case H.NOSCRIPT:
											case H.SCRIPT:
											case H.STYLE:
												f = u.flattenArrayTypeChildren({child: d, arrayTypeChildren: f, newChildProps: y, nestedChildren: g});
												break;
											default:
												a = u.mapObjectTypeChildren({child: d, newProps: a, newChildProps: y, nestedChildren: g});
												break;
										}
									}
								}),
								(a = this.mapArrayTypeChildrenToProps(f, a)),
								a
							);
						}),
						(i.prototype.render = function () {
							var l = this.props,
								a = l.children,
								u = yf(l, ['children']),
								f = Fe({}, u);
							return a && (f = this.mapChildrenToProps(a, f)), A.createElement(t, f);
						}),
						Kb(i, null, [
							{
								key: 'canUseDOM',
								set: function (l) {
									t.canUseDOM = l;
								},
							},
						]),
						i
					);
				})(A.Component)),
			(n.propTypes = {
				base: se.object,
				bodyAttributes: se.object,
				children: se.oneOfType([se.arrayOf(se.node), se.node]),
				defaultTitle: se.string,
				defer: se.bool,
				encodeSpecialCharacters: se.bool,
				htmlAttributes: se.object,
				link: se.arrayOf(se.object),
				meta: se.arrayOf(se.object),
				noscript: se.arrayOf(se.object),
				onChangeClientState: se.func,
				script: se.arrayOf(se.object),
				style: se.arrayOf(se.object),
				title: se.string,
				titleAttributes: se.object,
				titleTemplate: se.string,
			}),
			(n.defaultProps = {defer: !0, encodeSpecialCharacters: !0}),
			(n.peek = t.peek),
			(n.rewind = function () {
				var o = t.rewind();
				return o || (o = Tv({baseTag: [], bodyAttributes: {}, encodeSpecialCharacters: !0, htmlAttributes: {}, linkTags: [], metaTags: [], noscriptTags: [], scriptTags: [], styleTags: [], title: '', titleAttributes: {}})), o;
			}),
			r
		);
	},
	dk = function () {
		return null;
	},
	fk = Rb(Jb, rk, Tv)(dk),
	qa = ck(fk);
qa.renderStatic = qa.rewind;
const Pv = p.createContext({
		sessionId: null,
		setSessionId: () => null,
		agentId: null,
		setAgentId: () => null,
		newSession: null,
		setNewSession: () => null,
		sessions: [],
		setSessions: () => null,
		agents: [],
		setAgents: () => null,
		customers: [],
		setCustomers: () => null,
		openDialog: () => null,
		closeDialog: () => null,
	}),
	Do = () => p.useContext(Pv);
function pk() {
	const e = p.lazy(() => cw(() => import('./chat-BI4eCUxH.js'), __vite__mapDeps([0, 1]))),
		[t, n] = p.useState(null),
		[r, o] = p.useState([]),
		[i, s] = p.useState(null),
		[l, a] = p.useState(null),
		[u, f] = p.useState(''),
		[d, c] = p.useState([]),
		[g, w] = p.useState([]),
		{openDialog: y, DialogComponent: S, closeDialog: m} = yb();
	p.useEffect(() => {
		var x;
		if (t)
			if (t === dv) f('Parlant | New Session');
			else {
				const E = (x = r == null ? void 0 : r.find((C) => C.id === t)) == null ? void 0 : x.title;
				E && f(`Parlant | ${E}`);
			}
		else f('Parlant');
	}, [t, r]);
	const h = {sessionId: t, setSessionId: n, agentId: i, setAgentId: s, newSession: l, setNewSession: a, sessions: r, setSessions: o, agents: d, setAgents: c, customers: g, setCustomers: w, openDialog: y, closeDialog: m};
	return v.jsx(vb, {
		children: v.jsxs(Pv.Provider, {
			value: h,
			children: [
				v.jsx(qa, {defaultTitle: `${u}`}),
				v.jsxs('div', {
					'data-testid': 'chatbot',
					className: 'main bg-main h-screen flex flex-col',
					children: [
						v.jsx(mb, {}),
						v.jsxs('div', {
							className: 'flex justify-between flex-1 w-full overflow-auto flex-row',
							children: [
								v.jsx('div', {className: 'bg-white h-full pb-4 border-solid w-[332px] max-mobile:hidden', children: v.jsx(yv, {})}),
								v.jsx('div', {className: 'h-full w-[calc(100vw-332px)] max-w-[calc(100vw-332px)] max-[750px]:max-w-full max-[750px]:w-full ', children: t && v.jsx(p.Suspense, {children: v.jsx(e, {})})}),
							],
						}),
					],
				}),
				v.jsx(S, {}),
			],
		}),
	});
}
function mk() {
	return v.jsx('div', {children: v.jsx(pk, {})});
}
var Sf = ['light', 'dark'],
	hk = '(prefers-color-scheme: dark)',
	gk = p.createContext(void 0),
	vk = {setTheme: (e) => {}, themes: []},
	yk = () => {
		var e;
		return (e = p.useContext(gk)) != null ? e : vk;
	};
p.memo(({forcedTheme: e, storageKey: t, attribute: n, enableSystem: r, enableColorScheme: o, defaultTheme: i, value: s, attrs: l, nonce: a}) => {
	let u = i === 'system',
		f = n === 'class' ? `var d=document.documentElement,c=d.classList;${`c.remove(${l.map((w) => `'${w}'`).join(',')})`};` : `var d=document.documentElement,n='${n}',s='setAttribute';`,
		d = o ? (Sf.includes(i) && i ? `if(e==='light'||e==='dark'||!e)d.style.colorScheme=e||'${i}'` : "if(e==='light'||e==='dark')d.style.colorScheme=e") : '',
		c = (w, y = !1, S = !0) => {
			let m = s ? s[w] : w,
				h = y ? w + "|| ''" : `'${m}'`,
				x = '';
			return o && S && !y && Sf.includes(w) && (x += `d.style.colorScheme = '${w}';`), n === 'class' ? (y || m ? (x += `c.add(${h})`) : (x += 'null')) : m && (x += `d[s](n,${h})`), x;
		},
		g = e
			? `!function(){${f}${c(e)}}()`
			: r
			? `!function(){try{${f}var e=localStorage.getItem('${t}');if('system'===e||(!e&&${u})){var t='${hk}',m=window.matchMedia(t);if(m.media!==t||m.matches){${c('dark')}}else{${c('light')}}}else if(e){${s ? `var x=${JSON.stringify(s)};` : ''}${c(
					s ? 'x[e]' : 'e',
					!0
			  )}}${u ? '' : 'else{' + c(i, !1, !1) + '}'}${d}}catch(e){}}()`
			: `!function(){try{${f}var e=localStorage.getItem('${t}');if(e){${s ? `var x=${JSON.stringify(s)};` : ''}${c(s ? 'x[e]' : 'e', !0)}}else{${c(i, !1, !1)};}${d}}catch(t){}}();`;
	return p.createElement('script', {nonce: a, dangerouslySetInnerHTML: {__html: g}});
});
const wk = ({...e}) => {
	const {theme: t = 'system'} = yk();
	return v.jsx(hS, {
		theme: t,
		className: 'toaster group',
		toastOptions: {
			classNames: {
				toast: 'group toast group-[.toaster]:bg-background group-[.toaster]:text-foreground group-[.toaster]:border-border group-[.toaster]:shadow-lg',
				description: 'group-[.toast]:text-muted-foreground',
				actionButton: 'group-[.toast]:bg-primary group-[.toast]:text-primary-foreground',
				cancelButton: 'group-[.toast]:bg-muted group-[.toast]:text-muted-foreground',
			},
		},
		...e,
	});
};
Im(document.getElementById('root')).render(v.jsxs(p.StrictMode, {children: [v.jsx(mk, {}), v.jsx(wk, {richColors: !0, position: 'bottom-right'})]}));
export {Mo as B, Qr as J, $l as N, A as R, cs as T, xk as a, QC as b, he as c, Nl as d, YC as e, V1 as f, Ar as g, v as j, Ck as p, p as r, Ia as t, Do as u};
