from experiments.occlusion import roof_occlusion, llava_roof_occlusion


def run_occlusion_suite():
    tests = [
        ("Roof Occlusion Tests", roof_occlusion.main),
        ("LLaVA Occlusion Tests", llava_roof_occlusion.main),
    ]

    for name, func in tests:
        print(f"\nRunning {name}...")
        func()
