#import "wg21.typ" as wg21

#show: wg21.template.with(
  paper_number: "DXXXXR0",
  audience: "LEWG",
  title: [Tweak trivial relocation library primitives \[DRAFT\] ],
  authors: (
    (
      name: "David Sankel",
      institution: "Adobe",
      mail: "dsankel@adobe.com",
    ),
    (
      name: "Jon Bauman",
      institution: "Rust Foundation",
      mail: "jonbauman@rustfoundation.org",
    ),
  ),
  date: (
    year: 2025,
    month: "September",
    day: 30,
  ),
  abstract: [
      "Trivial Relocatability For C++ 26"@TrivialRelocatability introduced
      mechanisms for the identification and tagging of types whose objects can
      be "trivially" relocated from one memory address to another. It also
      included some standard library functions that perform this relocation.
      Useful as they are, these standard library functions are insufficient for
      important use cases such as `realloc` support, value representation
      serialization, and cross language interoperability. We propose completing
      the trivial relocation function set with the addition of a single function
      template, `std::restart_lifetime`, that addresses these unsupported use cases.

      #table(
        columns: (1fr,1fr),
        stroke: none,
        table.header[*Before*][*After*],
        table.hline(),
        [
        ```Cpp
        // A trivially relocatable, but not trivially
        // copyable, type.
        class Foo { /*...*/ };

        // Create a foo sequence with a single element using
        // Microsoft's specialized mimalloc allocator.
        void* foo_sequence_buffer =
          mi_malloc_aligned(sizeof(Foo), alignof(Foo));
        Foo* foo_sequence = new (foo_sequence_buffer) Foo();

        // Extend the sequence reusing the same memory if
        // possible
        foo_sequence_buffer = mi_realloc_aligned(
          foo_sequence, sizeof(Foo)*2, alignof(Foo));
        new (foo_sequence_buffer+sizeof(Foo)) Foo();
        foo_sequence = (Foo*)foo_sequence_buffer;




        foo_sequence[0].bar(); // Undefined behavior
        ```
        ],
        [
        ```Cpp









        // ...same as before...







        // Restart lifetime of relocated elements
        std::restart_lifetime(foo_sequence[0]);

        foo_sequence[0].bar(); // Okay
        ```
        ],
      )
  ],
)

// TODO: Consider another before/after table demonstrating improved Rust interop
// ergonomics.

= Introduction

It is a common-but-unspecified property that for many types, an object can be
relocated with a `memcpy` of its underlying bytes. Although the standard
guarantees this only for the small number of _trivially copyable_ types,
virtually all C++ compilers support `memcpy`-relocation of non-self-referential
types. Many applications have taken advantage of this property for performance
optimizations and a number of libraries have emerged that attempt to surface
this functionality in a generic way.
#footnote[See @TrivialRelocatability and @TrivialRelocatabilityAlt for a survey of such libraries.]

After much debate, a form of this functionality landed in the working
draft@CppStandardDraft with "Trivial Relocatability For C++
26"@TrivialRelocatability#footnote[See @TrivialRelocatabilityAlt for a notable
design alternative that was considered, but ultimately rejected.]. An important
trade-off in this design is that qualifying types may be "trivially" relocated
using only the `trivially_relocate` function; `memcpy` will not suffice.

```Cpp
// Foo is a trivially relocatable, but not
// trivially copyable, type.
class Foo { /*...*/ };
static_assert(
     std::is_trivially_relocatable_v<Foo>()
 && !std::is_trivially_copyable_v<Foo>());

void f() {
  alignas(Foo) char x1_buffer[sizeof(Foo)],
                    x2_buffer[sizeof(Foo)],
                    y1_buffer[sizeof(Foo)],
                    y2_buffer[sizeof(Foo)];

  // Relocating using std::memcpy results in
  // undefined behavior.
  Foo* x1 = new (x1_buffer) Foo();
  std::memcpy(&y1_buffer, x1, sizeof(Foo));
  Foo* y1 = reinterpret_cast<Foo*>(y1_buffer);
  y1->bar(); // Undefined behavior

  // Relocating using std::trivially_relocate
  // works as expected.
  Foo* x2 = new (x2_buffer) Foo();
  Foo* y2 = std::trivially_relocate(
    x2,
    x2+1,
    reinterpret_cast<Foo*>(&y2_buffer));
  y2->bar(); // Okay
}
```

One of the motivations for this trade-off is the ARM64e ABI which encodes an
object's address in its virtual table (vtable) pointer making
`memcpy`-relocation impossible for polymorphic types on this
platform#footnote[This is a memory safety vulnerability mitigation. See
@PointerAuthentication and @Arm64e for details.]. The requirement to call
`std::trivially_relocate` provides an opportunity for the standard library to
perform "fix ups" on these vtable pointers.

While `std::trivially_relocate` suffices for many use cases and neatly handles
the ARM64e platform, other important use-cases remain unaddressed. We propose to
compliment `std::trivially_relocate` with another function,
`std::restart_lifetime`, that addresses these use-cases and nicely compliments
`std::trivially_relocate`.

= Key `std::trivially_relocate` limitations

== `realloc` use case

// TODO: fill in. See mimalloc. Also https://github.com/rhempel/umm_malloc

== Serialization use case

// TODO: fill in

== Rust-interop use case

// TODO: fill in

= `start_lifetime_as`

// TODO: Create some examples that explain how start_lifetime_as would be used and demonstrate how it fixes things.

= Other considerations

// TODO: Fill this section. Review email thread to identify other discussions on this

== Will this cause security problems?

== Why is this being brought up now?

== Is this a bug fix or a feature?

== Is this critical for C++26?

= Alternatives considered

// TODO: Add the alternative of ripping out trivial relocatability from the standard.

== `start_lifetime_at` extension

// TODO: Add prose explaining this and why it was discarded. We can probably get away
// with showing use cases instead of standardese.

#wg21.standardese[
```
template<class T>
T* start_lifetime_at(uintptr_t origin,
                     void* p) noexcept;
```

_Mandates_: `is_trivially_relocatable_v<T> && !is_const_v<T>` is `true`.

_Preconditions_:
- [`p`, `(char*)p + sizeof(T)`) denotes a region of allocated storage that
  is a subset of the region of storage reachable through [basic.compound] `p`
  and suitably aligned for the type `T`.
- The contents of [`p`, `(char*)p + sizeof(T)`) is the value representation of
  an object `a` that was stored at `origin`.

_Effects_: Implicitly creates an object _b_ within the denoted region of type
`T` whose address is `p`, whose lifetime has begun, and whose object
representation is the same as that of _a_.

_Returns_: A pointer to the _b_ defined in the _Effects_ paragraph.
]

= Wording

// TODO: Create start_lifetime_as wording

= Conclusion

// TODO: Add conclusion

= Acknowledgments

// TODO: Mention Pablo Halpern and others who contributed to the conversation.

#bibliography("references.yml", style: "ieee")
